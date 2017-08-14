Open the jupyter notebook, open the Run.ipynb

Execute the commands in sequences
    main() executes the artistic style transfer using MRF prior with default
    parameters.

To provide customized parameters, use 
    main(style_weight='<weight>', content_file='<content file name>', style_file='<style file name>')

    where 1e0 is the weight of the style loss function, the content_file is the file name of the
    content source image and the style_file is the file name of the style source image.

    The weight of the content loss function is default to 2e0 and the weight of the
    total variation smooth regularization is 1e-3. User can customize the ratio of the
    weights between the content and the style by providing the style_weight with different values.

    The content_file and style_file must be in the "./data/" subdirectory, by default
    it uses ./data/face_content.jpg and ./data/face_style.jpg

    The input is cropped to 128x128 pixel, the user can further modify the input cropping pixel dimension
    by diving into the source code of MyStyleTranferMRF.py
