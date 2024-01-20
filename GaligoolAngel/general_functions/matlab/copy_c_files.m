function copy_c_files(sourcePath, destPath)
    % Ensure the paths end with file separators
    if ~endsWith(sourcePath, filesep)
        sourcePath = strcat(sourcePath, filesep);
    end

    if ~endsWith(destPath, filesep)
        destPath = strcat(destPath, filesep);
    end

    % Check if the source and destination folders exist
    if ~isfolder(sourcePath)
        error('Source folder does not exist.');
    end

    if ~isfolder(destPath)
        mkdir(destPath);
    end

    % Find all .c files in the source directory
    files = dir(fullfile(sourcePath, '*.c'));

    % Copy each file to the destination directory
    for i = 1:length(files)
        sourceFile = fullfile(sourcePath, files(i).name);
        destFile = fullfile(destPath, files(i).name);
        copyfile(sourceFile, destFile);
    end

    disp('All .c files have been copied.');
end
