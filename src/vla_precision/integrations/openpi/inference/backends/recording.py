# Episode recording shared by the native robot backends.
import threading
import queue
import yaml
import time
import cv2
import av
import numpy as np
from pathlib import Path
from .plots import save_episode_plot

class Recorder:
    """This class handles asynchronous logging and visualization."""

    def __init__(self, log_path: Path, video_path: list, display_fps: int = 15, visualize: bool = False):
        self.visualize = visualize
        # log path    
        self.log_path = log_path

        # video paths
        self.left_wrist_video = video_path[0]
        self.exterior_video = video_path[1]
        self.right_wrist_video = video_path[2] if len(video_path) > 2 else None

        self.display_fps = display_fps
        
        # safe queues
        self.queue_log = queue.Queue(maxsize=1)
        self.queue_vis = queue.Queue(maxsize=1)

        # store frames for video
        self.frames_ext = []    # exterior camera frames
        self.frames_left_wrist = []  # left wrist camera frames
        self.frames_right_wrist = []  # optional right wrist camera frames
        self.episode_results = []
        self._episodes_flushed = False

        # start threads
        threading.Thread(target=self._logger_thread, daemon=True).start()
        threading.Thread(target=self._visualizer_thread, daemon=True).start()

    # ====================== Logger Thread ====================== #
    def _logger_thread(self):
        while True:
            data = self.queue_log.get()  # blocking
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None, allow_unicode=True)
                    f.write(' ' + '-' * 60 + '\n')
                    f.flush()
            except Exception as e:
                print(f"[Recorder Logger ERROR] {e}")
            finally:
                self.queue_log.task_done()

    # ====================== Visualizer Thread ====================== #
    def _visualizer_thread(self):
        last_frame_time = 0
        while True:
            try:
                obs = self.queue_vis.get()
                if "observation/right_wrist_image" in obs:
                    l_wrist, r_wrist, ext = map(
                        self.to_bgr,
                        [
                            obs["observation/left_wrist_image"],
                            obs["observation/right_wrist_image"],
                            obs["observation/exterior_image"],
                        ],
                    )
                else:
                    l_wrist, ext = map(
                        self.to_bgr,
                        [obs["observation/wrist_image"], obs["observation/image"]],
                    )
                    r_wrist = None
                # save frames in memory (convert to uint8 to save space)
                self.frames_ext.append(ext.astype(np.uint8))
                self.frames_left_wrist.append(l_wrist.astype(np.uint8))
                if r_wrist is not None:
                    self.frames_right_wrist.append(r_wrist.astype(np.uint8))

                if self.visualize:
                    # concatenate and display
                    frames = [l_wrist, ext] if r_wrist is None else [l_wrist, r_wrist, ext]
                    combined = np.hstack(frames)
                    cv2.imshow(
                        "Left Wrist | Exterior" if r_wrist is None else "Left Wrist | Right Wrist | Exterior",
                        combined,
                    )
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # control frame rate
                elapsed = time.time() - last_frame_time
                if elapsed < 1 / self.display_fps:
                    time.sleep(1 / self.display_fps - elapsed)
                last_frame_time = time.time()
            except Exception as e:
                print(f"[VIS ERROR] {e}")
                time.sleep(0.1)
    
    def _encode(self, frames, out_path: Path, vcodec: str = "libx264", crf: int = 23, preset="medium"):
        h, w, _ = frames[0].shape
        container = av.open(str(out_path), "w")
        stream = container.add_stream(vcodec, rate=self.display_fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": str(preset)}

        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        # flush
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"{out_path.name}: {size_mb:.2f} MB")
        return size_mb
        
    # ====================== Save Video ====================== #
    def save_video(self):
        """Save stored frames as two separate MP4 videos using H.264."""
        if not self.frames_ext or not self.frames_left_wrist:
            print("No frames to save.")
            return

        print("\nSaving exterior camera video...")
        self._encode(self.frames_ext, self.exterior_video, vcodec="libx264", crf=23, preset="veryslow")

        print("\nSaving left wrist camera video...")
        self._encode(self.frames_left_wrist, self.left_wrist_video, vcodec="libx264", crf=23, preset="veryslow")

        if self.right_wrist_video is not None:
            if not self.frames_right_wrist:
                print("No right wrist frames to save.")
            else:
                print("\nSaving right wrist camera video...")
                self._encode(
                    self.frames_right_wrist,
                    self.right_wrist_video,
                    vcodec="libx264",
                    crf=23,
                    preset="veryslow",
                )

    # ====================== Save Video ====================== #
    def save_videos_multi_codec(self):
        """Save frames in memory as videos with H.264 / H.265 / AV1."""

        if not self.frames_ext or not self.frames_left_wrist:
            print("No frames to save.")
            return

        # ---------------- Helper ---------------- #
        def _encode(frames, out_path: Path, vcodec: str, crf: int = 30, preset="medium"):
            h, w, _ = frames[0].shape
            container = av.open(str(out_path), "w")
            stream = container.add_stream(vcodec, rate=self.display_fps)
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"

            # Set options
            if vcodec in ["libx264", "libx265", "libsvtav1"]:
                stream.options = {"crf": str(crf), "preset": str(preset)}

            for frame in frames:
                video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            # flush
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"🎞️ {out_path.name}: {size_mb:.2f} MB")
            return size_mb

        codecs = [
            ("libx264", 18, "veryslow"),
            ("libx265", 28, "slow"),
            ("libsvtav1", 35, "8"),  # preset 8 = slower
        ]

        print("\n🔧 Saving exterior camera videos...")
        for vcodec, crf, preset in codecs:
            out_path = self.exterior_video.parent / f"ext_{vcodec}.mp4"
            _encode(self.frames_ext, out_path, vcodec=vcodec, crf=crf, preset=preset)

        print("\n🔧 Saving left wrist camera videos...")
        for vcodec, crf, preset in codecs:
            out_path = self.left_wrist_video.parent / f"left_wrist_{vcodec}.mp4"
            _encode(self.frames_left_wrist, out_path, vcodec=vcodec, crf=crf, preset=preset)

    # ====================== Utility Functions ====================== #
    def to_bgr(self, img):
        """Convert RGB → BGR safely."""
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    # ====================== External Interface ====================== #
    def _make_actions_record(self, actions, infer_time: int, prompt: str = "", state=None, **extra):
        try:
            actions_array = np.asarray(actions, dtype=np.float64)
            # Keep enough precision to compare raw actions with idle thresholds.
            actions_list = np.round(actions_array, 6).tolist()
        except Exception:
            actions_array = actions
            try:
                actions_list = list(actions)
            except Exception:
                actions_list = actions

        delta_list = []
        for i, row in enumerate(actions_array):
            if i == 0:
                delta = row.copy()
            else:
                delta = row - actions_array[i - 1]
            delta_list.append(np.round(delta[:6], 6).tolist())

        data = dict(extra)
        data["infer_time"] = int(infer_time)
        data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        data["prompt"] = prompt
        if state is not None:
            try:
                data["state"] = np.round(np.asarray(state, dtype=np.float64).reshape(-1), 6).tolist()
            except Exception:
                data["state"] = list(state)
        # Keep state immediately above actions in the emitted YAML record.
        data["actions"] = actions_list
        data["actions_delta"] = delta_list
        if isinstance(actions_array, np.ndarray) and actions_array.ndim == 2 and actions_array.shape[1] >= 14:
            right_delta_list = []
            for i, row in enumerate(actions_array):
                delta = row.copy() if i == 0 else row - actions_array[i - 1]
                right_delta_list.append(np.round(delta[7:13], 6).tolist())
            data["actions_delta_left"] = delta_list
            data["actions_delta_right"] = right_delta_list
        return data

    def submit_actions(self, actions, infer_time: int, prompt: str = "", state=None, **extra):
        """Submit action results only."""
        data = self._make_actions_record(actions, infer_time, prompt, state=state, **extra)
        try:
            self.queue_log.put_nowait(data)
        except queue.Full:
            pass
        return data

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        total_seconds = int(round(seconds))
        hours, rem = divmod(total_seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes}m {secs}s"
        if minutes:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    def _make_episode_result(
        self,
        episode_index: int | None = None,
        duration_sec=None,
        status: str = "target_reached",
        action_batches: int | None = None,
        completed: bool | None = None,
        **extra,
    ):
        if episode_index is None:
            episode_index = len(self.episode_results) + 1

        if completed is None:
            completed = status == "target_reached"

        result = {
            "episode_index": int(episode_index),
            "status": status,
            "completed": bool(completed),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if action_batches is not None:
            result["action_batches"] = int(action_batches)

        if duration_sec is None:
            result.update({"duration_sec": None, "duration": ""})
        else:
            duration_sec = float(duration_sec)
            result.update({
                "duration_sec": round(duration_sec, 3),
                "duration": self._format_duration(duration_sec),
            })
        result.update(extra)
        return result

    def _store_episode_result(self, episode_result: dict):
        self.episode_results.append(episode_result)
        if episode_result["duration_sec"] is None:
            print(f"[EPISODE] Episode {episode_result['episode_index']} {episode_result['status']}: incomplete")
        else:
            print(
                f"[EPISODE] Episode {episode_result['episode_index']} {episode_result['status']}: "
                f"{episode_result['duration']}"
            )

    def submit_episode_result(
        self,
        episode_index: int | None = None,
        duration_sec=None,
        status: str = "target_reached",
        action_batches: int | None = None,
        completed: bool | None = None,
        **extra,
    ):
        """Submit episode result only."""
        result = self._make_episode_result(episode_index, duration_sec, status, action_batches, completed, **extra)
        self._store_episode_result(result)
        return result

    def submit_episode_summary(self, **extra):
        """Submit final episode timing summary and print it to terminal."""
        completed = [
            item
            for item in self.episode_results
            if item.get("duration_sec") is not None and item.get("completed")
        ]
        incomplete = [item for item in self.episode_results if item not in completed]
        durations = [item["duration_sec"] for item in completed]
        avg_duration = float(np.mean(durations)) if durations else 0.0
        total_episodes = len(self.episode_results)
        success_rate = len(completed) / total_episodes if total_episodes else 0.0
        summary = {
            "total_episodes": total_episodes,
            "completed_episodes": len(completed),
            "incomplete_episodes": len(incomplete),
            "success_rate": round(success_rate, 4),
            "success_rate_percent": round(success_rate * 100.0, 2),
            "average_completed_duration_sec": round(avg_duration, 3),
            "average_completed_duration": self._format_duration(avg_duration),
            "episodes": list(self.episode_results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        summary.update(extra)
        plot_path = save_episode_plot(summary, log_path=self.log_path)
        if plot_path is not None:
            summary["episode_plot"] = plot_path

        if not self._episodes_flushed:
            for item in self.episode_results:
                self.queue_log.put({"episode_result": item})
                self.queue_log.join()
            self._episodes_flushed = True

        self.queue_log.put({"episode_summary": summary})
        self.queue_log.join()

        print("========== Episode Timing Summary ==========")
        if self.episode_results:
            print(
                f"[EPISODE] Total {len(self.episode_results)} episodes, "
                f"completed {len(completed)}, incomplete {len(incomplete)}, "
                f"success rate {summary['success_rate_percent']:.2f}%"
            )
            print(
                f"[EPISODE] Average over {len(completed)} completed episodes: "
                f"{summary['average_completed_duration']} "
                f"({summary['average_completed_duration_sec']}s)"
            )
            if summary.get("episode_plot"):
                print(f"[EPISODE] Plot saved to: {summary['episode_plot']}")
        else:
            print("[EPISODE] No submitted action episodes.")
        return summary

    def submit_obs(self, obs: dict):
        """
        obs dictionary should contain at least:
        - observation/image: exterior camera image (numpy array)
        - observation/wrist_image: wrist camera image (numpy array)

        Dual-arm observations instead use observation/exterior_image,
        observation/left_wrist_image, and observation/right_wrist_image.
        """
        try:
            self.queue_vis.put_nowait(obs)
        except queue.Full:
            pass
