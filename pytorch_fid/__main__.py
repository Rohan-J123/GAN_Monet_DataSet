if __name__ == "__main__":
    import fid_score
    s = ["/content/GAN_Monet_DataSet/generated_jpg_1", "/content/GAN_Monet_DataSet/generated_jpg_2", "/content/GAN_Monet_DataSet/generated_jpg_3", "/content/GAN_Monet_DataSet/generated_jpg_4", "/content/GAN_Monet_DataSet/generated_jpg_5", "/content/GAN_Monet_DataSet/generated_jpg_6", "/content/GAN_Monet_DataSet/generated_jpg_7", "/content/GAN_Monet_DataSet/generated_jpg_8"]
    for one_s in s:
        fid_score.main(one_s)
