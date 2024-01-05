def encode_settings(encode_method, new_width, new_height, fps, output, ffmpeg_path, sharpen, sharpen_sens):
    command = [ffmpeg_path,
               '-y',
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-s', f'{new_width}x{new_height}',
               '-pix_fmt', 'rgb24',
               '-r', str(fps),
               '-i', '-',
               '-an']
    
    # Settings from: https://www.xaymar.com/guides/obs/high-quality-recording/
    # X265 is broken for some reason, I don't know why just yet
    # TO-DO: Add AMF Support
    match encode_method:
        case "x264":
            command.extend(['-c:v', 'libx264',
                            '-preset', 'veryfast',
                            '-crf', '15',
                            output])

        case "x264_animation":
            command.extend(['-c:v', 'libx264',
                            '-preset', 'veryfast',
                            '-tune', 'animation',
                            '-crf', '15',
                            output])
            
        case "x265":
            command.extend(['-c:v', 'libx265',
                            '-preset', 'veryfast',
                            '-crf', '15',
                            output])
        
        case "x265_amimation":
            command.extend(['-c:v', 'libx265',
                            '-preset', 'veryfast',
                            '-tune', 'animation',
                            '-crf', '15',
                            output])
            
        case "nvenc_h264":
            command.extend(['-c:v', 'h264_nvenc',
                            '-preset', 'p1',
                            '-cq', '15',
                            output])
            
        case "nvenc_h265":
            command.extend(['-c:v', 'hevc_nvenc',
                            '-preset', 'p1',
                            '-cq', '15',
                            output])
            
        case "qsv_h264":
            command.extend(['-c:v', 'h264_qsv',
                            '-preset', 'veryfast',
                            '-global_quality', '15',
                            output])
            
        case "qsv_h265":
            command.extend(['-c:v', 'hevc_qsv',
                            '-preset', 'veryfast',
                            '-global_quality', '15',
                            output])

    if sharpen:
        command.insert(-1, '-vf')
        command.insert(-1, f'cas={sharpen_sens}')

    return command
