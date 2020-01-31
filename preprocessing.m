clc;clear;

infolder = './data/global/';
outfolder = './data/local/';

mkdir(outfolder);

vfov = 90;
headmove_h = (0:10:80);
headmove_v = (0:10:80);

imlist=dir(infolder);
parfor n=3:length(imlist)
    fileName = imlist(n).name
    im360= imread([infolder '/' fileName ]);
    im360 = imresize(im360, [1024 2048]);

    file_idx = regexp(fileName, 'P', 'split');
    file_idx = regexp(cell2mat(file_idx(2)), '\.', 'split');
    file_idx = cell2mat(file_idx(1));

    iml = size(im360,1);
    imw = size(im360,2);

    for hh = 1:length(headmove_h)
        offset=round(headmove_h(hh)/360*imw);
        im_turned = [im360(:,imw-offset+1:imw,:) im360(:,1:imw-offset,:)];
        for hv = 1:length(headmove_v)
            [out] = equi2cubic(im_turned, iml, vfov, headmove_v(hv));
            for i=1:6
                imwrite(cell2mat(out(i)), [outfolder 'P' file_idx '_' num2str(hv) '_' num2str(hh) '_' num2str(i) '.jpg']);
            end
        end
    end
end
