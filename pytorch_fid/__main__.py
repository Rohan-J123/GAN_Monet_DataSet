if __name__ == "__main__":
    import fid_score
    s = ["/content/GAN_Monet_DataSet/generated_jpg_8"]
    for one_s in s:
        fid_score.main(one_s)
