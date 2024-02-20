function maxNum = findLastFileInFolder(folderPath, template)
    % List all files in the folder
    files = dir(folderPath);
    fileNames = {files.name};
    
    % Initialize maximum number
    maxNum = -inf;
    
    % Regular expression to match the pattern 'smthX' and extract 'X'
    pattern = [template, '(\d+)'];
    
    % Loop through each file name
    for i = 1:length(fileNames)
        % Extract the number from the file name
        matches = regexp(fileNames{i}, pattern, 'tokens');
        if ~isempty(matches)
            num = str2double(matches{1});
            % Update maxNum if this number is larger
            if num > maxNum
                maxNum = num;
            end
        end
    end
    
    % Handle case where no matching files were found
    if isinf(maxNum)
        maxNum = NaN; % Indicates no valid files were found
        disp('No files matching the pattern were found.');
    else
        disp(['The largest X found is: ', num2str(maxNum)]);
    end
end
