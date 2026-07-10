def iterBatches(read_frame, batch_size: int):
    while True:
        frames = []
        for _ in range(batch_size):
            frame = read_frame()
            if frame is None:
                break
            frames.append(frame)
        if not frames:
            return
        yield frames
        if len(frames) < batch_size:
            return
