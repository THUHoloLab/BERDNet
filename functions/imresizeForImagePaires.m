function dataOut = imresizeForImagePaires(dataIn,outputSize)


dataOut = cell(size(dataIn));
for idx = 1:size(dataIn,1)

    % Image resize
    dataOut(idx,:) = {
        im2single(imresize(dataIn{idx,1},outputSize)), ...
        im2single(imresize(dataIn{idx,2},outputSize))};

end

end