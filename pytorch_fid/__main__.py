if __name__ == "__main__":
    import fid_score
    # s = ["../generated_jpg_1", "../generated_jpg_2", "../generated_jpg_3", "../generated_jpg_4", "../generated_jpg_5", "../generated_jpg_6", "../generated_jpg_7"]
    s = ["../generated_jpg_8"]
    for one_s in s:
        fid_score.main(one_s)
