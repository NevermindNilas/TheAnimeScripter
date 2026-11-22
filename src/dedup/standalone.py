"""``--dedup``: drop duplicate frames, as a run of its own.

Decode, ask the detector about each frame, write the ones it keeps. The video
gets shorter and the frame rate stays, so audio is off (the validator turns it
off). It is not a stage of the processing chain: ``--smooth_dedup`` is the
duplicate handling that combines with interpolation, and it keeps the duration.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from src.constants import ADOBE
from src.infra.progressBarLogic import ProgressBarLogic
from src.io.ffmpegSettings import (
    BuildBuffer,
    closeWriterAndDrainReader,
    createWriteBuffer,
)
from src.io.runOutcome import recordWriterFailure, truncatedDecodeError

if ADOBE:
    from src.server.aeComms import progressState, reportTerminalStatus


class DedupVideo:
    def __init__(
        self,
        input,
        output,
        width,
        height,
        fps,
        detector,
        inpoint=0,
        outpoint=0,
        encode_method="x264",
        benchmark=False,
        totalFrames=0,
        bitDepth: str = "8bit",
        half: bool = True,
        decodeMethod: str = "cpu",
        decodeResize: bool = False,
    ):
        self.input = input
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        # Called once per decoded frame, in order; truthy means "duplicate of
        # the last frame kept" (src/factories/dedup.py).
        self.detector = detector
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.encode_method = encode_method
        self.benchmark = benchmark
        self.totalFrames = totalFrames
        self.bitDepth = bitDepth
        self.half = half
        self.decodeMethod = decodeMethod
        # width/height are a --resize target the decoder scales to.
        self.decodeResize = decodeResize

        self.consumed = 0
        self.dropped = 0
        # Read back by the standalone factory: this path never reaches start(),
        # so it is how main.py counts a failed run as failed.
        self.processingError: Exception | None = None
        try:
            self.run()
        except Exception as e:
            self.processingError = e
            logging.exception(f"Something went wrong while deduplicating, {e}")

        # Every thread has finished: a late encoder error counts too.
        recordWriterFailure(self)
        if ADOBE:
            reportTerminalStatus(self.processingError, self.output, self.benchmark)

    def run(self):
        if ADOBE:
            progressState.update({"status": "Removing duplicate frames..."})

        self.readBuffer = BuildBuffer(
            videoInput=self.input,
            inpoint=self.inpoint,
            outpoint=self.outpoint,
            half=self.half,
            resize=self.decodeResize,
            width=self.width,
            height=self.height,
            bitDepth=self.bitDepth,
            decode_method=self.decodeMethod,
        )
        self.writeBuffer = createWriteBuffer(
            input=self.input,
            output=self.output,
            encode_method=self.encode_method,
            width=self.width,
            height=self.height,
            fps=self.fps,
            grayscale=False,
            transparent=False,
            benchmark=self.benchmark,
            bitDepth=self.bitDepth,
            inpoint=self.inpoint,
            outpoint=self.outpoint,
        )

        with ThreadPoolExecutor(max_workers=3) as executor:
            writeFuture = executor.submit(self.writeBuffer)
            decodeFuture = executor.submit(self.readBuffer)
            processFuture = executor.submit(self.process)
            decodeFuture.result()
            processFuture.result()
            writeFuture.result()

    def process(self):
        try:
            with ProgressBarLogic(
                self.totalFrames,
                title=os.path.splitext(os.path.basename(self.input))[0],
                outputPath=self.output,
                videoFps=self.fps,
            ) as bar:
                read = self.readBuffer.read
                write = self.writeBuffer.write
                detector = self.detector
                while (frame := read()) is not None:
                    self.consumed += 1
                    # The bar tracks decoded frames, so dropped ones advance it.
                    bar(1)
                    if detector(frame):
                        self.dropped += 1
                        bar.setStats(
                            f"{self.dropped} dupes removed", f"{self.dropped} dup"
                        )
                        continue
                    # The decoder's frame is finished work and goes on as it
                    # is, so the writer's private stream has nothing to wait on.
                    write(frame)
                if self.consumed != self.totalFrames:
                    bar.updateTotal(self.consumed)
        finally:
            # The decoder signals a mid-stream death only by its end-of-stream
            # sentinel, which reads like a clean EOF.
            decodeError = truncatedDecodeError(
                self.readBuffer, self.totalFrames, self.writeBuffer
            )
            if decodeError is not None and self.processingError is None:
                self.processingError = decodeError
            # Also on a raise: otherwise the reader stays blocked on put() and
            # the executor join never returns.
            closeWriterAndDrainReader(self.writeBuffer, self.readBuffer)
            logging.info(f"Processed {self.consumed} frames")
            if self.dropped:
                logging.info(f"Deduplicated {self.dropped} frames")
