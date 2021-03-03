clear all;
close all;

orig = h5read('orig.h5','/image');
zf = h5read('regrid.h5','/image');
rec = h5read('recon.h5','/recon');

fid = fopen('cs.bin','rb');
fista = fread(fid,'float32');
fclose(fid);
fista = reshape(fista(1:2:end)+1j*fista(2:2:end),140,140,28);
fista = max(abs(orig(:)))*fista/abs(max(fista(:)));

orig = permute(orig,[3 2 1]);
zf = permute(zf,[3 2 1]);
rec = permute(rec,[3 2 1]);

figure; montage(reshape(orig,140,140,1,28),'DisplayRange',[0 0.5]); title('Original');
figure; montage(reshape(zf,140,140,1,28),'DisplayRange',[0 0.5]); title('Gridding');
figure; montage(reshape(rec,140,140,1,28),'DisplayRange',[0 0.5]); title('CNN');

figure; montage(reshape(abs(orig-zf),140,140,1,28),'DisplayRange',[0 0.1]); title('Orig-ZF');
figure; montage(reshape(abs(orig-rec),140,140,1,28),'DisplayRange',[0 0.1]); title('Orig-CNN');

fista = max(abs(orig(:)))*fista/abs(fista(:));

orig8 = orig(:,:,8);
zf8 = zf(:,:,8);
rec8 = rec(:,:,8);
fista8 = fista(:,:,8);

figure; imshow(imrotate(orig8,-90),[]); imshow(imrotate(zf8,-90),[]); imshow(imrotate(rec8,-90),[]); imshow(imrotate(fista8,-90),[]);

imshow(abs(orig8-zf8),[0 0.1]); imshow(abs(orig8-rec8),[0 0.1]); imshow(abs(orig8-fista8),[0 0.1]);