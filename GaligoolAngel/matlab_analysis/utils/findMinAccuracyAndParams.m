function [minAcc, minParams] = findMinAccuracyAndParams(filename)
    % Open the file
    fid = fopen(filename, 'rt');
    if fid == -1
        error('Cannot open file: %s', filename);
    end
    
    minAcc = inf;
    minParams = '';
    currentParams = '';
    
    % Read the file line by line
    while ~feof(fid)
        line = fgetl(fid);
        
        % Extract parameters line
        if contains(line, 'Time taken for hyperparameters:')
            currentParams = line;
        end
        
        % Check if the line contains the accuracy information
        if contains(line, 'Mean dev acc for taken hyperparameters:')
            % Extract the number using regular expressions
            numStr = regexp(line, '\d+\.\d+', 'match');
            acc = str2double(numStr{1});
            % Update the minimum accuracy and parameters if the current one is smaller
            if acc < minAcc
                minAcc = acc;
                minParams = currentParams;
            end
        end
    end
    
    % Close the file
    fclose(fid);
end
