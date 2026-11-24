def iterBatches(read_frame, batch_size: int):
    """Yield lists of up to ``batch_size`` frames until ``read_frame`` returns None.

    The reader sends a single end marker and blocks on any read after it, so
    nothing is read once the marker has been seen.
    """
    while True:
        batch = []
        while len(batch) < batch_size:
            frame = read_frame()
            if frame is None:
                if batch:
                    yield batch
                return
            batch.append(frame)
        yield batch
