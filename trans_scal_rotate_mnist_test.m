clc;
clear all;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Image to matrix conversion%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data_path='/home/sidistic/MNIST_data/mnist_png/testing';

d=dir(data_path);
folders=char(d.name);
folders=folders(3:end,:);
kk=1;
for i=1:size(folders,1)
    
 filepath=strcat(data_path,'/',folders(i,:));
 
 dd=dir(filepath);
 files=char(dd.name);
 files=files(3:end,:);
 for j=1:size(files,1)
 image_path=strcat(filepath,'/',files(j,:));
 
 image(kk,:,:)=imread(image_path);
 label(kk)=i-1;
    
    kk=kk+1;
    
    
 end
    
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Random location generation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rand_loc=randi(10000,3000,1);

rand_loc_trans=rand_loc(1:1000);
rand_loc_rot=rand_loc(1001:2000);
rand_scal=rand_loc(2001:3000);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img_rot=image(rand_loc_rot,:,:);
 r = randi([-60 60],1000,1);
 for ii=1:1000
     
     img_rot_final(ii,:,:)=imresize(imrotate(reshape(img_rot(ii,:,:),[28,28]),r(ii)),[28,28]);
     
 end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 img_scal=image(rand_scal,:,:);
 
 s = randi([30 98],1000,1)./100;
 for ii=1:1000
     
     img_scal_final(ii,:,:)=imresize(imresize(reshape(img_scal(ii,:,:),[28,28]),s(ii)),[28,28]);
     
 end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img_trans=image(rand_loc_trans,:,:);

t=randi([0 4],1000,2);

  for ii=1:1000
     
     img_trans_final(ii,:,:)=imresize(imtranslate(reshape(img_trans(ii,:,:),[28,28]),[t(ii,1),t(ii,2)]),[28,28]);
     
 end




subplot(211)
imshow(reshape(image(7350,:,:),[28,28]));
subplot(212)
imshow(reshape(modified_image(7350,:,:),[28,28]));
%imshow(J)



modified_image=image;
modified_image(rand_loc_rot,:,:)=img_rot_final;
modified_image(rand_loc_trans,:,:)=img_trans_final;
modified_image(rand_scal,:,:)=img_scal_final;


save('mnist_modified_image_test.mat','modified_image');%modified image
save('mnist_image_test.mat','image');                   %orginal image
save('mnist_image_label_test.mat','label');              %label files
save('mnist_modified_rand_loc_test.mat','rand_loc');    % random loacations









% imshow(reshape(image(end,:,:),[28,28]));