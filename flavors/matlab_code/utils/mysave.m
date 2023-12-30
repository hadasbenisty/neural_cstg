function mysave(f,filename, ext)
if isempty(f)
    f=gcf;
end
[pth, name] = fileparts(filename);
mkNewFolder(pth);
mkNewFolder(fullfile(pth, 'fig'));
mkNewFolder(fullfile(pth, 'eps'));
mkNewFolder(fullfile(pth, 'tif'));
if ~exist('ext','var') || isempty(ext)
    saveas(f,fullfile(pth, 'fig', [name '.fig']),'fig');
    saveas(f,fullfile(pth, 'tif', [name '.tif']),'tif');
    set(f, 'Renderer', 'painters');
    print( f, '-depsc', fullfile(pth, 'eps', [name '.eps']));
else
    switch ext
        case 'tif'
            fullfile(pth, 'tif', [name '.tif'])
        case 'fig'
            fullfile(pth, 'fig', [name '.fig'])
        case 'eps'
            set(f, 'Renderer', 'painters');
            fullfile(pth, 'eps', [name '.eps'])
    end
end

% print(f,[filename '.pdf'],'-dpdf');
