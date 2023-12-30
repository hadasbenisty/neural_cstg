function animals_db = get_animals_list(datapath, animals_names)

for animal_i = 1:length(animals_names)
    animals_db{animal_i} = readtable(fullfile(datapath, 'animals_db_selected.xlsx'),'Sheet',animals_names{animal_i});
%     inds = strcmp(animals_db{animal_i}.codition, 'control') & ...
%         (strcmp(animals_db{animal_i}.type1, 'ongoing') | ...
%         strcmp(animals_db{animal_i}.type1, 'first')) & ...
%         strcmp(animals_db{animal_i}.type2, 'batch');
%     animals_db{animal_i} = animals_db{animal_i}(inds,:);
end




