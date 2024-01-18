function relPath = calculateRelativePath(source, target)
    % Split the paths into cell arrays of folders
    sourceParts = strsplit(source, filesep);
    targetParts = strsplit(target, filesep);

    % Find the common path
    minLength = min(length(sourceParts), length(targetParts));
    commonLength = 0;
    for i = 1:minLength
        if strcmp(sourceParts{i}, targetParts{i})
            commonLength = i;
        else
            break;
        end
    end

    % Calculate the steps to go up from the source path to the common path
    upSteps = length(sourceParts) - commonLength;
    upPath = repmat({'..'}, 1, upSteps);

    % Append the unique part of the target path
    uniqueTarget = targetParts(commonLength+1:end);
    relPathArray = [upPath, uniqueTarget];

    % Join the parts to form the relative path
    relPath = strjoin(relPathArray, filesep);
    if isempty(relPath)
        relPath = '.';
    end
end

