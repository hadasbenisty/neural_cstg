function mkNewFolder(pth)

if isempty(pth)
    return;
end
if ~exist(pth, 'dir')
    mkdir(pth);
end
    