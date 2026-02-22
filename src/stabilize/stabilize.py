import logging
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from src.constants import ADOBE
from src.stabilize.superpoint import SuperPoint, find_match_index, find_transform
import src.utils.ffmpegSettings as ffmpegSettings
from src.utils.ffmpegSettings import BuildBuffer, WriteBuffer
from src.utils.progressBarLogic import ProgressBarLogic

if ADOBE:
    from src.utils.aeComms import progressState


class VideoStabilize:
    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        custom_encoder="",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "8bit",
    ):
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.bitDepth = bitDepth

        self.smoothingSeconds = 2.0
        self.smoothingMethod = "grad_opt"
        self.optimizationIterations = 60
        self.maxCorners = 200
        self.qualityLevel = 0.01
        self.minDistance = 30
        self.flowWinSize = (21, 21)
        self.angleMaxHard = 45.0
        self.analysisWidth = 640
        self.orbFeatures = 3000
        self.orbScaleFactor = 1.2
        self.orbLevels = 8
        self.orbRatioTest = 0.8
        self.superPointMatchThreshold = 0.3
        self.superPointMaxKeypoints = 4096
        self.superPointDetectionThreshold = 0.01
        self.superPointModel = None
        self.superPointDevice = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.superPointOptimizationIterations = 50

        self.transforms = []
        self.meanMatchScores = []
        self.shiftXFix = None
        self.shiftYFix = None
        self.angleFix = None

        self.orb = cv2.ORB_create(
            nfeatures=self.orbFeatures,
            scaleFactor=self.orbScaleFactor,
            nlevels=self.orbLevels,
            edgeThreshold=31,
            patchSize=31,
            fastThreshold=15,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._initSuperPoint()

        try:
            self.runStreamingPipeline()
        except Exception as e:
            logging.exception(f"Something went wrong in stabilization pipeline, {e}")

    def runStreamingPipeline(self):
        if ADOBE:
            progressState.update({"status": "Analyzing and stabilizing video..."})

        with ProgressBarLogic(self.totalFrames * 2, title=self.input) as bar:
            self.analyzeMotion(progressBar=bar, advance=1)
            self.computeTrajectoryCorrection()
            self.renderStabilized(progressBar=bar, advance=1)

    def _clearReaderCache(self):
        try:
            ffmpegSettings.CachedReader = None
            ffmpegSettings.CachedReaderMethod = None
        except Exception as e:
            logging.warning(f"Failed to clear cached decoder reader: {e}")

    def analyzeMotion(self, progressBar=None, advance=1):
        if ADOBE:
            progressState.update({"status": "Analyzing camera motion for stabilization..."})

        self._clearReaderCache()

        self.readBuffer = BuildBuffer(
            videoInput=self.input,
            inpoint=self.inpoint,
            outpoint=self.outpoint,
            resize=False,
            width=self.width,
            height=self.height,
            toTorch=False,
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            decodeFuture = executor.submit(self.readBuffer)
            analyzeFuture = executor.submit(self._analyzeFrames, progressBar, advance)
            decodeFuture.result()
            analyzeFuture.result()

        self.transforms = np.array(self.transforms, dtype=np.float32)
        self.meanMatchScores = np.array(self.meanMatchScores, dtype=np.float32)

        logging.info(
            f"Stabilization analysis complete: {len(self.transforms)} transforms, {len(self.meanMatchScores)} match scores"
        )

    def _initSuperPoint(self):
        try:
            self.superPointModel = (
                SuperPoint(
                    nms_radius=4,
                    max_num_keypoints=self.superPointMaxKeypoints,
                    detection_threshold=self.superPointDetectionThreshold,
                    remove_borders=4,
                    descriptor_dim=256,
                    channels=[64, 64, 128, 128, 256],
                )
                .load(map_location=self.superPointDevice)
                .eval()
                .to(self.superPointDevice)
            )
            logging.info(f"SuperPoint initialized on {self.superPointDevice}")
        except Exception as e:
            self.superPointModel = None
            logging.warning(
                f"SuperPoint initialization failed, falling back to ORB/LK ({e})"
            )

    def _grayToSuperPointTensor(self, gray):
        h, w = gray.shape[:2]
        maxDim = max(h, w)
        resizeScale = 1.0

        if maxDim > self.analysisWidth:
            resizeScale = self.analysisWidth / float(maxDim)
            newW = max(32, int(round(w * resizeScale)))
            newH = max(32, int(round(h * resizeScale)))
            gray = cv2.resize(gray, (newW, newH), interpolation=cv2.INTER_AREA)

        tensor = (
            torch.from_numpy(gray)
            .to(self.superPointDevice, dtype=torch.float32)
            .div(255.0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return tensor, resizeScale

    def _estimateTransform(self, prevGray, currGray):
        superPointResult = self._estimateTransformSuperPoint(prevGray, currGray)
        if superPointResult is not None:
            return superPointResult

        orbResult = self._estimateTransformORB(prevGray, currGray)
        if orbResult is not None:
            return orbResult

        prevPts = cv2.goodFeaturesToTrack(
            prevGray,
            maxCorners=self.maxCorners,
            qualityLevel=self.qualityLevel,
            minDistance=self.minDistance,
            blockSize=3,
        )

        if prevPts is None or len(prevPts) < 6:
            return 0.0, 0.0, 0.0, 0.0

        currPts, status, error = cv2.calcOpticalFlowPyrLK(
            prevGray,
            currGray,
            prevPts,
            None,
            winSize=self.flowWinSize,
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if currPts is None or status is None:
            return 0.0, 0.0, 0.0, 0.0

        valid = status.reshape(-1) == 1
        if np.count_nonzero(valid) < 6:
            return 0.0, 0.0, 0.0, 0.0

        prevMatched = prevPts[valid].reshape(-1, 2)
        currMatched = currPts[valid].reshape(-1, 2)

        matrix, _ = cv2.estimateAffinePartial2D(
            prevMatched,
            currMatched,
            method=cv2.RANSAC,
            ransacReprojThreshold=3,
        )

        if matrix is None:
            return 0.0, 0.0, 0.0, 0.0

        dx = float(matrix[0, 2])
        dy = float(matrix[1, 2])
        da = float(np.arctan2(matrix[1, 0], matrix[0, 0]))

        if error is None:
            matchScore = 0.0
        else:
            err = error.reshape(-1)[valid]
            if err.size == 0:
                matchScore = 0.0
            else:
                medianErr = float(np.median(err))
                matchScore = float(np.exp(-medianErr / 10.0))
                matchScore = float(np.clip(matchScore, 0.0, 1.0))

        return dx, dy, da, matchScore

    def _estimateTransformSuperPoint(self, prevGray, currGray):
        if self.superPointModel is None:
            return None

        prevTensor, resizeScale = self._grayToSuperPointTensor(prevGray)
        currTensor, _ = self._grayToSuperPointTensor(currGray)

        try:
            autocastCtx = (
                torch.autocast(device_type=self.superPointDevice.type)
                if self.superPointDevice.type != "cpu"
                else nullcontext()
            )
            with torch.inference_mode(), autocastCtx:
                kpBatch = self.superPointModel.infer(torch.cat([prevTensor, currTensor], dim=0))

            kp1 = kpBatch[0]
            kp2 = kpBatch[1]
            if kp1["keypoints"].shape[0] < 10 or kp2["keypoints"].shape[0] < 10:
                return None

            idx1, idx2, scores = find_match_index(
                kp1,
                kp2,
                threshold=self.superPointMatchThreshold,
                return_score=True,
            )

            if idx1.numel() < 10:
                return None

            pts1Torch = kp1["keypoints"][idx1].detach().float()
            pts2Torch = kp2["keypoints"][idx2].detach().float()
            scoreVals = scores.detach().float().cpu().numpy()
            center = [float(prevTensor.shape[3]) * 0.5, float(prevTensor.shape[2]) * 0.5]

            try:
                shift, _, angle, _ = find_transform(
                    pts1Torch,
                    pts2Torch,
                    center=center,
                    iteration=self.superPointOptimizationIterations,
                    sigma=2.0,
                    disable_scale=True,
                )
                dx = float(shift[0].item())
                dy = float(shift[1].item())
                da = float(angle.item())
                inlierRatio = 1.0
            except Exception:
                pts1 = pts1Torch.cpu().numpy()
                pts2 = pts2Torch.cpu().numpy()
                matrix, inliers = cv2.estimateAffinePartial2D(
                    pts1,
                    pts2,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=2.5,
                    confidence=0.995,
                )
                if matrix is None:
                    return None

                dx = float(matrix[0, 2])
                dy = float(matrix[1, 2])
                da = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
                if inliers is None:
                    inlierRatio = 0.0
                else:
                    inlierRatio = float(np.mean(inliers.astype(np.float32)))

            if resizeScale != 1.0:
                invScale = 1.0 / resizeScale
                dx *= invScale
                dy *= invScale

            descriptorScore = float(np.mean(scoreVals)) if scoreVals.size > 0 else 0.0
            matchScore = float(
                np.clip(0.6 * descriptorScore + 0.4 * inlierRatio, 0.0, 1.0)
            )
            return dx, dy, da, matchScore
        except Exception as e:
            logging.warning(f"SuperPoint estimation failed, using ORB/LK fallback ({e})")
            return None

    def _estimateTransformORB(self, prevGray, currGray):
        prevH, prevW = prevGray.shape[:2]
        maxDim = max(prevH, prevW)
        resizeScale = 1.0

        if maxDim > self.analysisWidth:
            resizeScale = self.analysisWidth / float(maxDim)
            newW = max(32, int(round(prevW * resizeScale)))
            newH = max(32, int(round(prevH * resizeScale)))
            prevWork = cv2.resize(prevGray, (newW, newH), interpolation=cv2.INTER_AREA)
            currWork = cv2.resize(currGray, (newW, newH), interpolation=cv2.INTER_AREA)
        else:
            prevWork = prevGray
            currWork = currGray

        kp1, des1 = self.orb.detectAndCompute(prevWork, None)
        kp2, des2 = self.orb.detectAndCompute(currWork, None)

        if (
            des1 is None
            or des2 is None
            or kp1 is None
            or kp2 is None
            or len(kp1) < 10
            or len(kp2) < 10
        ):
            return None

        try:
            knnMatches = self.matcher.knnMatch(des1, des2, k=2)
        except Exception:
            return None

        goodMatches = []
        for pair in knnMatches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.orbRatioTest * n.distance:
                goodMatches.append(m)

        if len(goodMatches) < 10:
            return None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in goodMatches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in goodMatches])

        matrix, inliers = cv2.estimateAffinePartial2D(
            pts1,
            pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,
            confidence=0.995,
        )

        if matrix is None:
            return None

        dx = float(matrix[0, 2])
        dy = float(matrix[1, 2])
        da = float(np.arctan2(matrix[1, 0], matrix[0, 0]))

        if resizeScale != 1.0:
            invScale = 1.0 / resizeScale
            dx *= invScale
            dy *= invScale

        if inliers is None:
            inlierRatio = 0.0
        else:
            inlierRatio = float(np.mean(inliers.astype(np.float32)))

        if len(goodMatches) > 0:
            meanDistance = float(np.mean([m.distance for m in goodMatches]))
        else:
            meanDistance = 256.0

        distanceScore = float(np.exp(-meanDistance / 64.0))
        matchScore = float(np.clip(0.7 * inlierRatio + 0.3 * distanceScore, 0.0, 1.0))

        return dx, dy, da, matchScore

    def _analyzeFrames(self, progressBar=None, advance=1):
        frameCount = 0
        prevGray = None

        for _ in range(self.totalFrames):
            frame = self.readBuffer.read()
            if frame is None:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            if prevGray is None:
                self.transforms.append((0.0, 0.0, 0.0))
                self.meanMatchScores.append(0.0)
            else:
                dx, dy, da, score = self._estimateTransform(prevGray, gray)
                self.transforms.append((dx, dy, da))
                self.meanMatchScores.append(score)

            prevGray = gray
            frameCount += 1
            if progressBar is not None:
                progressBar(advance)

            if self.readBuffer.isReadFinished() and self.readBuffer.isQueueEmpty():
                break

        logging.info(f"Analyzed {frameCount} frames for stabilization")

    def _calcSceneWeight(self, meanMatchScores):
        if meanMatchScores.size == 0:
            return meanMatchScores

        maxScore = 0.75
        minScore = 0.50
        weight = (meanMatchScores - minScore) / (maxScore - minScore)
        weight = np.clip(weight, 0.0, 1.0)
        lowMask = weight < 0.65
        weight[lowMask] = weight[lowMask] ** 2

        weight[0] = 0.0
        weight[-1] = 0.0
        return weight.astype(np.float32)

    def _calcSceneWeightScalar(self, matchScore: float):
        maxScore = 0.75
        minScore = 0.50
        weight = (float(matchScore) - minScore) / (maxScore - minScore)
        weight = float(np.clip(weight, 0.0, 1.0))
        if weight < 0.65:
            weight = weight**2
        return weight

    def _movingAverage(self, curve, radius):
        windowSize = 2 * radius + 1
        if len(curve) < windowSize:
            return curve

        weights = np.ones(windowSize, dtype=np.float32) / windowSize
        padded = np.pad(curve, (radius, radius), mode="edge")
        smoothed = np.convolve(padded, weights, mode="same")
        return smoothed[radius:-radius]

    def _convSmoothing(self, shiftX, shiftY, angle):
        if shiftX.shape[0] < 3:
            return shiftX, shiftY, angle

        window = int(round(self.fps * self.smoothingSeconds))
        if window < 3:
            window = 3
        if window % 2 == 0:
            window += 1
        radius = window // 2

        shiftXSmooth = self._movingAverage(shiftX, radius)
        shiftYSmooth = self._movingAverage(shiftY, radius)
        angleSmooth = self._movingAverage(angle, radius)
        return shiftXSmooth, shiftYSmooth, angleSmooth

    def _gradOptSmoothing(self, tx, ty, ta, sceneWeight):
        if tx.shape[0] < 8:
            return tx, ty, ta

        device = "cpu"

        txTensor = torch.tensor(tx, dtype=torch.float64, device=device)
        tyTensor = torch.tensor(ty, dtype=torch.float64, device=device)
        taTensor = torch.tensor(ta, dtype=torch.float64, device=device)
        swTensor = torch.tensor(sceneWeight, dtype=torch.float64, device=device)

        px = txTensor.clone().requires_grad_(True)
        py = tyTensor.clone().requires_grad_(True)
        pa = taTensor.clone().requires_grad_(True)

        gradWeight = 1.0
        penaltyWeight = 2e-3 / max(self.smoothingSeconds, 0.25)

        optimizer = torch.optim.LBFGS(
            [px, py, pa],
            history_size=10,
            max_iter=4,
            line_search_fn="strong_wolfe",
        )

        swSafe = swTensor

        def closure():
            optimizer.zero_grad()
            loss = torch.tensor(0.0, dtype=torch.float64, device=device)

            for pred, target in ((px, txTensor), (py, tyTensor), (pa, taTensor)):
                fx1 = pred[1:] - pred[:-1]
                fx2 = fx1[1:] - fx1[:-1]
                fx3 = fx2[1:] - fx2[:-1]

                gradLoss = torch.tensor(0.0, dtype=torch.float64, device=device)
                if fx1.numel() > 0:
                    gradLoss = gradLoss + (fx1.pow(2) * swSafe[: fx1.shape[0]]).mean()
                if fx2.numel() > 0:
                    gradLoss = gradLoss + (fx2.pow(2) * swSafe[: fx2.shape[0]]).mean()
                if fx3.numel() > 0:
                    gradLoss = gradLoss + (fx3.pow(2) * swSafe[: fx3.shape[0]]).mean()

                penalty = (pred - target).pow(2).mean()
                loss = loss + gradLoss * gradWeight + penalty * penaltyWeight

            loss.backward()
            return loss

        for _ in range(self.optimizationIterations):
            optimizer.step(closure)

        return (
            px.detach().cpu().numpy(),
            py.detach().cpu().numpy(),
            pa.detach().cpu().numpy(),
        )

    def computeTrajectoryCorrection(self):
        if self.transforms.size == 0:
            self.shiftXFix = np.zeros((0,), dtype=np.float32)
            self.shiftYFix = np.zeros((0,), dtype=np.float32)
            self.angleFix = np.zeros((0,), dtype=np.float32)
            return

        shiftX = self.transforms[:, 0].astype(np.float64)
        shiftY = self.transforms[:, 1].astype(np.float64)
        angle = self.transforms[:, 2].astype(np.float64)

        angleDeg = np.degrees(angle)
        angleDeg = np.clip(angleDeg, -self.angleMaxHard, self.angleMaxHard)
        angle = np.radians(angleDeg)

        sceneWeight = self._calcSceneWeight(self.meanMatchScores.astype(np.float64))

        shiftXWeighted = np.cumsum(shiftX * sceneWeight)
        shiftYWeighted = np.cumsum(shiftY * sceneWeight)
        angleWeighted = np.cumsum(angle * sceneWeight)

        if self.smoothingMethod == "grad_opt":
            shiftXSmooth, shiftYSmooth, angleSmooth = self._gradOptSmoothing(
                shiftXWeighted,
                shiftYWeighted,
                angleWeighted,
                sceneWeight.astype(np.float64),
            )
        else:
            shiftXSmooth, shiftYSmooth, angleSmooth = self._convSmoothing(
                shiftXWeighted,
                shiftYWeighted,
                angleWeighted,
            )

        self.shiftXFix = (shiftXSmooth - shiftXWeighted).astype(np.float32)
        self.shiftYFix = (shiftYSmooth - shiftYWeighted).astype(np.float32)
        self.angleFix = (angleSmooth - angleWeighted).astype(np.float32)

        logging.info(
            f"Computed trajectory correction for {self.shiftXFix.shape[0]} frames (method={self.smoothingMethod})"
        )

    def _buildAffineFromCorrection(self, dx, dy, da, centerX, centerY):
        cosA = np.cos(da)
        sinA = np.sin(da)
        return np.array(
            [
                [cosA, -sinA, (1.0 - cosA) * centerX + sinA * centerY + dx],
                [sinA, cosA, (1.0 - cosA) * centerY - sinA * centerX + dy],
            ],
            dtype=np.float32,
        )

    def _processStreamingFrames(self):
        frameCount = 0
        prevGray = None

        cumulativeX = 0.0
        cumulativeY = 0.0
        cumulativeA = 0.0

        smoothedX = 0.0
        smoothedY = 0.0
        smoothedA = 0.0

        centerX = self.width / 2.0
        centerY = self.height / 2.0

        fpsSafe = max(float(self.fps), 1.0)
        smoothingFrames = max(self.smoothingSeconds * fpsSafe, 1.0)
        alpha = 1.0 / smoothingFrames

        try:
            with ProgressBarLogic(self.totalFrames) as bar:
                for _ in range(self.totalFrames):
                    frame = self.readBuffer.read()
                    if frame is None:
                        break

                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                    if prevGray is not None:
                        dx, dy, da, score = self._estimateTransform(prevGray, gray)

                        daDeg = np.degrees(da)
                        daDeg = np.clip(daDeg, -self.angleMaxHard, self.angleMaxHard)
                        da = np.radians(daDeg)

                        sceneWeight = self._calcSceneWeightScalar(score)

                        cumulativeX += dx * sceneWeight
                        cumulativeY += dy * sceneWeight
                        cumulativeA += da * sceneWeight

                        smoothedX += alpha * (cumulativeX - smoothedX)
                        smoothedY += alpha * (cumulativeY - smoothedY)
                        smoothedA += alpha * (cumulativeA - smoothedA)

                        fixX = smoothedX - cumulativeX
                        fixY = smoothedY - cumulativeY
                        fixA = smoothedA - cumulativeA
                    else:
                        fixX, fixY, fixA = (0.0, 0.0, 0.0)

                    transform = self._buildAffineFromCorrection(
                        fixX,
                        fixY,
                        fixA,
                        centerX,
                        centerY,
                    )

                    stabilized = cv2.warpAffine(
                        frame,
                        transform,
                        (self.width, self.height),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )

                    outputTensor = (
                        torch.from_numpy(stabilized)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .float()
                        .div(255.0)
                    )
                    self.writeBuffer.write(outputTensor)

                    prevGray = gray
                    frameCount += 1
                    bar(1)

                    if self.readBuffer.isReadFinished() and self.readBuffer.isQueueEmpty():
                        break

            logging.info(f"Processed {frameCount} frames (single-pass stabilize)")
        finally:
            self.writeBuffer.close()

    def renderStabilized(self, progressBar=None, advance=1):
        if ADOBE:
            progressState.update({"status": "Applying stabilization and encoding video..."})

        self._clearReaderCache()

        self.readBuffer = BuildBuffer(
            videoInput=self.input,
            inpoint=self.inpoint,
            outpoint=self.outpoint,
            resize=False,
            width=self.width,
            height=self.height,
            toTorch=False,
        )

        self.writeBuffer = WriteBuffer(
            self.input,
            self.output,
            self.encode_method,
            self.custom_encoder,
            self.width,
            self.height,
            self.fps,
            sharpen=False,
            sharpen_sens=None,
            grayscale=False,
            benchmark=self.benchmark,
            bitDepth=self.bitDepth,
        )

        with ThreadPoolExecutor(max_workers=3) as executor:
            writeFuture = executor.submit(self.writeBuffer)
            decodeFuture = executor.submit(self.readBuffer)
            renderFuture = executor.submit(self._renderFrames, progressBar, advance)
            decodeFuture.result()
            renderFuture.result()
            writeFuture.result()

    def _renderFrames(self, progressBar=None, advance=1):
        frameCount = 0
        try:
            centerX = self.width / 2.0
            centerY = self.height / 2.0

            for i in range(self.totalFrames):
                frame = self.readBuffer.read()
                if frame is None:
                    break

                if self.shiftXFix is not None and i < self.shiftXFix.shape[0]:
                    dx = float(self.shiftXFix[i])
                    dy = float(self.shiftYFix[i])
                    da = float(self.angleFix[i])
                else:
                    dx, dy, da = (0.0, 0.0, 0.0)

                transform = self._buildAffineFromCorrection(dx, dy, da, centerX, centerY)

                stabilized = cv2.warpAffine(
                    frame,
                    transform,
                    (self.width, self.height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )

                outputTensor = (
                    torch.from_numpy(stabilized)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .div(255.0)
                )

                self.writeBuffer.write(outputTensor)
                frameCount += 1
                if progressBar is not None:
                    progressBar(advance)

                if self.readBuffer.isReadFinished() and self.readBuffer.isQueueEmpty():
                    break

            logging.info(f"Processed {frameCount} frames")
        finally:
            self.writeBuffer.close()
