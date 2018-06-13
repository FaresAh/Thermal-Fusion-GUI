prompt = 'Enter the image number : ';
number = input(prompt, 's');

%path_to_ir = getIrPath(str2double(number));
path_to_ir = 'raw_image/';
path_to_rgb = 'rgb/';

path_to_rgb = strcat(path_to_rgb, 'IMG_');

imageRGB = strcat(path_to_rgb, number, '.jpg');
imageIR = strcat(path_to_ir, number, '_old.png');

fixed = imread(imageRGB);
moving = imread(imageIR);

[optimizer, metric] = imregconfig('multimodal');

tformSimilarity = imregister(rgb2gray(moving), rgb2gray(fixed),'rigid',optimizer,metric);

%this is the image we want
registered = tformSimilarity;

figure; 
subplot(2, 2, 1), imshow(registered), title('Registered image')
subplot(2, 2, 2), imshow(moving), title('IR image')
subplot(2, 2, 3), imshowpair(moving,registered,'blend'), title('superposition of both images')
subplot(2, 2, 4), imshow(fixed), title('RGB image')

cropped_path = strcat('cropped/test', number, '.jpg');
imwrite(registered, cropped_path)