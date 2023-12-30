x0=0.5+find(diff((BehaveData.sucrose.indicatorPerTrial))~=0);
figure;h(1)=subplot(4,1,1);imagesc(squeeze(pcaTrajres.projeff(1,:,:)))
for i=1:length(x0)    
   line([1 1]*x0(i), get(gca, 'YLim'), 'Color','k'); 
end
h(2)=subplot(4,1,2);imagesc(squeeze(pcaTrajres.projeff(2,:,:)))
for i=1:length(x0)    
   line([1 1]*x0(i), get(gca, 'YLim'), 'Color','k'); 
end
h(3)=subplot(4,1,3);imagesc(squeeze(pcaTrajres.projeff(3,:,:)))
for i=1:length(x0)    
   line([1 1]*x0(i), get(gca, 'YLim'), 'Color','k'); 
end
h(4)=subplot(4,1,4);plot(labels)
colormap jet;
linkaxes(h,'x')

[~,ic]=sort(labels);
figure;h(1)=subplot(4,1,1);imagesc(squeeze(pcaTrajres.projeff(1,:,ic)))

h(2)=subplot(4,1,2);imagesc(squeeze(pcaTrajres.projeff(2,:,ic)))

h(3)=subplot(4,1,3);imagesc(squeeze(pcaTrajres.projeff(3,:,ic)))

h(4)=subplot(4,1,4);plot(labels(ic))
colormap jet;
linkaxes(h,'x')
