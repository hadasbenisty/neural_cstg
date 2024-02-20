function paramsStruct = buildParamsStruct(line)
    % Extract the parameter part of the line, avoiding capturing the time
    paramPart = regexp(line, '.*:', 'match', 'once');

    % Define patterns for matching parameters
    % This pattern is for general parameter extraction
    generalPattern = '(?<name>\w+)\s*=\s*(?<value>[^,]+?)(?=[,:\s]|$)';

    % Find matches for the general pattern
    matches = regexp(paramPart, generalPattern, 'names');

    % Initialize an empty struct for parameters
    paramsStruct = struct();

    % Process each matched parameter
    for i = 1:length(matches)
        paramName = matches(i).name; % Extract parameter name
        paramValueStr = matches(i).value; % Extract parameter value as string

        % Attempt to interpret parameter values
        % Check if the value is numeric or an array of numerics
        if contains(paramValueStr, '[')
            % It's an array, remove brackets and split by comma
            numericValues = str2num(paramValueStr(2:end-1)); %#ok<ST2NM>
            paramsStruct.(paramName) = numericValues;
        else
            % Attempt to convert string to numeric value
            numericValue = str2double(paramValueStr);
            if isnan(numericValue) % Retain as string if conversion fails
                paramsStruct.(paramName) = paramValueStr;
            else
                paramsStruct.(paramName) = numericValue;
            end
        end
    end
end
