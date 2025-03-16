import unittest
import subprocess
import time


class TestCommandExecution(unittest.TestCase):
    def test_command_execution_failure(self):
        encoders = [
            "x264",
            "slow_x264",
            "x264_10bit",
            "x264_animation",
            "x264_animation_10bit",
            "x265",
            "slow_x265",
            "x265_10bit",
            "nvenc_h264",
            "slow_nvenc_h264",
            "nvenc_h265",
            "slow_nvenc_h265",
            "nvenc_h265_10bit",
            "nvenc_av1",
            "slow_nvenc_av1",
            "qsv_h264",
            "qsv_h265",
            "qsv_h265_10bit",
            "av1",
            "slow_av1",
            "h264_amf",
            "hevc_amf",
            "hevc_amf_10bit",
            "prores",
            "prores_segment",
            "gif",
            "vp9",
            "qsv_vp9",
            "lossless",
            "lossless_nvenc",
        ]

        failed = []
        success = []

        for encoder in encoders:
            print(f"Testing encoder: {encoder}")
            # Use a separate try/except for each potential point of failure
            cmd = [
                "python",
                "f:\\TheAnimeScripter\\main.py",
                "--input",
                "f:\\TheAnimeScripter\\input\\1080.mp4",
                "--interpolate",
                "--interpolate_method",
                "dedup",
                "--outpoint",
                "2",
                "--encode_method",
                encoder,
            ]

            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    errors="replace",
                )

                try:
                    stdout, stderr = process.communicate(timeout=10)

                    if (
                        process.returncode != 0
                        or "Broken pipe" in stdout
                        or "Broken pipe" in stderr
                        or "Error during encoding" in stdout
                        or "Error during encoding" in stderr
                        or "[Errno 32]" in stdout
                        or "[Errno 32]" in stderr
                    ):
                        failed.append(encoder)
                        print(f"  Failed: Return code {process.returncode}")

                        if "Broken pipe" in stdout or "Broken pipe" in stderr:
                            print("Found 'Broken pipe' in output")
                        if (
                            "Error during encoding" in stdout
                            or "Error during encoding" in stderr
                        ):
                            print("Found 'Error during encoding' in output")
                        if "[Errno 32]" in stdout or "[Errno 32]" in stderr:
                            print("Found '[Errno 32]' in output")
                    else:
                        success.append(encoder)
                        print("Succeeded")

                except subprocess.TimeoutExpired:
                    process.kill()
                    failed.append(encoder)
                    print("Failed: Process timed out after 10 seconds")

            except Exception as e:
                failed.append(encoder)
                print(f"  Failed: Unexpected error - {str(e)}")

            time.sleep(0.5)

        print("\n--- Test Summary ---")
        print(f"Failed encoders: {len(failed)}/{len(encoders)}")
        print(f"Successful encoders: {len(success)}/{len(encoders)}")

        if success:
            self.fail(f"The following encoders succeeded: {success}")
        else:
            print("All encoders failed as expected")


if __name__ == "__main__":
    unittest.main()
