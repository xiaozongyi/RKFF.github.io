# RKFF.github.io
# RKFF architecture for efficient sematic segmentation 
Our module starts from the second atrous convolution, with the input of atrous convolution fuses the result feature maps and the input feature maps of the last atrous convolution, so as to achieve the effect of "reviewing" local features. In addition, the RKFF core module also performs average pooling all the input. Finally, for the purpose of keeping the size of the feature maps, the billinear interpolation is followed the average pooling.

![fig08](https://user-images.githubusercontent.com/107088415/221349920-ac18e353-053b-4de1-90ec-4d1e197d68c2.jpg)

The segmentation results are as Figure

![fig14](https://user-images.githubusercontent.com/107088415/221349928-533c1eee-8516-4b9f-860b-79e93ab3ef21.jpg)

# Dataset
Both datasets used in this paper and required for training, validation, and testing can be downloaded directly from the dataset websites below:
Pascal VOC:http://host.robots.ox.ac.uk/pascal/VOC/
