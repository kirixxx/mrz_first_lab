from NeuralCompressor import NeuralCompressor
from ImgProcces import ImgProcces
from utils import save_compress_image


def main():
    img_procces = ImgProcces()

    img_arr = img_procces.img_to_array(img_path='images/mono.jpg')


    neural_compressor = NeuralCompressor(
        p=32,
        err=20000,
        a=0.0001,
        img_arr=img_arr
    )

    compress_matrix = neural_compressor.compress_img(
        pre_trained_neurons=False, 
        pre_trained_neurons_name='mono', 
        compressed_file_name='compressed/mono.npy'
    )
    save_compress_image('compressed/mono.npy', compress_matrix)
    decompress_matrix = neural_compressor.decompress_img(
        pre_trained_neurons_name='mono',
        compressed_file_name='compressed/mono.npy'
    ) 


    img_procces.array_to_img(decompress_matrix)


if __name__ == '__main__':
    main()
