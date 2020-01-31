clc;clear;

global_folder = './output/global/';
local_folder = './output/local/';
outfolder = './output/fused/';
mkdir(outfolder);

headmove_h = (0:10:80);
headmove_v = (0:10:80);
vfov = 90;
fusion_para = 0.55;

imlist = dir(global_folder);
parfor n = 3:length(imlist)
    fileName = imlist(n).name
    file_idx = regexp(fileName, 'P', 'split');
    file_idx = regexp(cell2mat(file_idx(2)), '\.', 'split');
    file_idx = cell2mat(file_idx(1));

    iml = size(imread([local_folder, 'P' file_idx '_1_1_1.jpg']), 1);

    im_cub_sal = zeros(iml, iml, 6, 3);
    Lsalmap = zeros(iml,iml*2);

    for hv = 1:length(headmove_v)
        for hh = 1:length(headmove_h)
            for i=1:6
                cubsal = double(imread([local_folder 'P' file_idx '_' num2str(hv) '_' num2str(hh) '_' num2str(i) '.jpg']));
                im_cub_sal(:,:,i,1) = cubsal;
                im_cub_sal(:,:,i,2) = cubsal;
                im_cub_sal(:,:,i,3) = cubsal;
            end
            im_tmp = cubic2equi(0,im_cub_sal(:,:,5,:), im_cub_sal(:,:,6,:), im_cub_sal(:,:,4,:), im_cub_sal(:,:,2,:), im_cub_sal(:,:,1,:), im_cub_sal(:,:,3,:));
            out = equi2cubic(im_tmp, iml, vfov, -headmove_v(hv));
            im_tmp = cubic2equi(-headmove_h(hh),cell2mat(out(5)),cell2mat(out(6)),cell2mat(out(4)),cell2mat(out(2)),cell2mat(out(1)),cell2mat(out(3)));
            im_tmp = im_tmp(:,:,1);
            im_tmp = double(im_tmp)+Lsalmap;
            Lsalmap = im_tmp;
        end
    end
    Lsalmap = Lsalmap./(hh*hv);
    Lsalmap = Lsalmap/max(max(Lsalmap));

    Gsalmap= double(imread([global_folder '/' fileName ]));

    iml = size(Gsalmap,1);
    imw = size(Gsalmap,2);

    Lsalmap = imresize(Lsalmap, [iml imw]);
    Lsalmap = double(Lsalmap./max(Lsalmap(:)).*255);

    Fsalmap = fusion_para.*Gsalmap+(1-fusion_para).*Lsalmap;
    Fsalmap = Fsalmap./max(Fsalmap(:))*255;
    imwrite(uint8(Fsalmap),[outfolder '/P' file_idx '.jpg']);
end
fprintf('Done');

