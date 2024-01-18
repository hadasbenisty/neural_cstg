function cent = get_centrality(C)
Np = size(C, 1);
cent.degree = nan(Np, size(C, 3));
cent.closeness = nan(Np, size(C, 3));
cent.betweenness = nan(Np, size(C, 3));
cent.pagerank = nan(Np, size(C, 3));
cent.eigenvector = nan(Np, size(C, 3));

for k = 1:size(C, 3)
    if all(all(~isnan(C(:,:,k))))
        G = graph(C(:,:,k));
        cent.degree(:, k) = centrality(G, 'degree','Importance', abs(G.Edges.Weight.^2))/Np;
        cent.closeness(:, k) = centrality(G, 'closeness','Cost', 1./abs(G.Edges.Weight.^2))/Np;
        cent.betweenness(:, k) = centrality(G, 'betweenness','Cost', 1./abs(G.Edges.Weight.^2))/Np;
        cent.pagerank(:, k) = centrality(G, 'pagerank','Importance', abs(G.Edges.Weight.^2));
        cent.eigenvector(:, k) = centrality(G, 'eigenvector','Importance', abs(G.Edges.Weight.^2))/Np;
    end
    
end