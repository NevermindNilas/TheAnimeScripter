"""
WORK IN PROGRESS, DO NOT USE YET!!!!!!!!!!!!!!

FOR FUTURE REFENRENCE FOR ME!

╰─ $body = @{
>     args = "--input ./input/1080.mp4 --dedup --benchmark"
> } | ConvertTo-Json
╰─ Invoke-RestMethod -Uri "http://localhost:5000/process" -Method Post -Body $body -ContentType "application/json"

ATTEMPT AT MAKING A TAS API SERVER USING FLASK, THIS IS HILARIOUS!
"""

from flask import Flask, request, jsonify
import sys
import os
import threading
import time
import logging
from platform import system
from signal import signal, SIGINT, SIG_DFL
import warnings

import src.constants as cs
from src.utils.coloredPrints import green
from src.utils.argumentsChecker import createParser
from src.utils.inputOutputHandler import processInputOutputPaths
from main import VideoProcessor

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.logger.setLevel(logging.ERROR)

currentJob = {
    "active": False,
    "jobId": None,
    "status": None,
    "args": None,
    "files": [],
    "currentFile": None,
    "processingTime": None,
    "timestamp": None,
    "outputFiles": [],
    "error": None,
    "progress": 0,
    "completedAt": None,
}

jobLock = threading.Lock()


def initializeEnvironment():
    cs.SYSTEM = system()
    cs.MAINPATH = (
        os.path.join(os.getenv("APPDATA"), "TheAnimeScripter")
        if cs.SYSTEM == "Windows"
        else os.path.join(
            os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
            "TheAnimeScripter",
        )
    )
    cs.WHEREAMIRUNFROM = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(cs.MAINPATH, exist_ok=True)

    # Setup logging
    signal(SIGINT, SIG_DFL)
    logging.basicConfig(
        filename=os.path.join(cs.MAINPATH, "TAS-Log.log"),
        filemode="w",
        format="%(message)s",
        level=logging.INFO,
    )
    logging.info("============== Flask Server Started ==============")


@app.route("/process", methods=["POST"])
def processVideo():
    """Process a video with the given arguments"""
    try:
        with jobLock:
            if currentJob["active"]:
                return jsonify(
                    {
                        "error": "A job is already running. Please wait for it to complete.",
                        "currentJobId": currentJob["jobId"],
                    }
                ), 409

        if not request.is_json:
            return jsonify({"error": "Invalid request format, expecting JSON"}), 400

        cmdArgs = request.json.get("args", "")
        if not cmdArgs:
            return jsonify({"error": "No arguments provided"}), 400

        jobId = str(int(time.time()))

        logging.info(f"Received process request with args: {cmdArgs}")

        argv = cmdArgs.split()

        isFrozen = hasattr(sys, "_MEIPASS")
        baseOutputPath = (
            os.path.dirname(sys.executable)
            if isFrozen
            else os.path.dirname(os.path.abspath(__file__))
        )

        oldArgv = sys.argv
        sys.argv = ["main.py"] + argv
        args = createParser(baseOutputPath)
        sys.argv = oldArgv

        outputPath = os.path.join(baseOutputPath, "output")
        results = processInputOutputPaths(args, outputPath)

        if not results:
            return jsonify({"error": "No videos found to process"}), 400

        def processThread():
            try:
                for i in results:
                    with jobLock:
                        currentJob["status"] = "processing"
                        currentJob["currentFile"] = results[i]["videoPath"]
                        currentJob["progress"] = 0

                    logging.info(f"Processing file: {results[i]['videoPath']}")

                    startTime = time.time()
                    VideoProcessor(args, results=results[i])
                    endTime = time.time()

                    with jobLock:
                        currentJob["processingTime"] = f"{endTime - startTime:.2f}"

                with jobLock:
                    currentJob["status"] = "completed"
                    currentJob["completedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")

                logging.info(f"Job {jobId} completed successfully")
            except Exception as e:
                with jobLock:
                    currentJob["status"] = "failed"
                    currentJob["error"] = str(e)

                logging.exception(f"Error in job {jobId}: {str(e)}")

        with jobLock:
            currentJob.update(
                {
                    "active": True,
                    "jobId": jobId,
                    "status": "starting",
                    "args": cmdArgs,
                    "files": [results[i]["videoPath"] for i in results],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "outputFiles": [results[i]["outputPath"] for i in results],
                    "error": None,
                    "progress": 0,
                    "completedAt": None,
                    "processingTime": None,
                    "currentFile": None,
                }
            )

        thread = threading.Thread(target=processThread)
        thread.daemon = True
        thread.start()

        return jsonify(
            {
                "jobId": jobId,
                "message": "Processing started",
                "files": [results[i]["videoPath"] for i in results],
            }
        )
    except Exception as e:
        logging.exception(f"Error starting job: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/status", methods=["GET"])
def getStatus():
    """Get the status of the current job"""
    with jobLock:
        if not currentJob["active"]:
            return jsonify({"status": "no_active_job"}), 404

        response = currentJob.copy()
        jobStatus = currentJob["status"]

        if jobStatus in ["completed", "failed"]:
            currentJob["active"] = False

    if jobStatus == "completed":
        return jsonify(response), 200
    elif jobStatus == "failed":
        return jsonify(response), 500
    else:
        return jsonify(response), 202


@app.route("/cancel", methods=["POST"])
def cancelJob():
    """Cancel the current job"""
    with jobLock:
        if not currentJob["active"]:
            return jsonify({"error": "No active job to cancel"}), 404

        if currentJob["status"] in ["completed", "failed"]:
            return jsonify(
                {"error": "Cannot cancel a job that is already finished"}
            ), 400

        currentJob["status"] = "cancelled"

    return jsonify({"message": "Job marked for cancellation"})


@app.route("/", methods=["GET"])
def index():
    """API home page"""
    return jsonify(
        {
            "name": "TheAnimeScripter API",
            "version": "1.0.0",
            "endpoints": [
                {
                    "url": "/process",
                    "method": "POST",
                    "description": "Process a video with given arguments",
                    "body": {"args": "command line arguments as string"},
                },
                {
                    "url": "/status",
                    "method": "GET",
                    "description": "Get status of the current job",
                },
                {
                    "url": "/cancel",
                    "method": "POST",
                    "description": "Cancel the current job",
                },
            ],
        }
    )


@app.errorhandler(404)
def notFound(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def serverError(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    initializeEnvironment()
    print(green("Starting TheAnimeScripter API server on http://0.0.0.0:5000"))
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
