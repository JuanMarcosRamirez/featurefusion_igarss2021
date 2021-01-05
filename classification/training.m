function [tran, test] = training (labs, Nt)  
    %% Dictionary of classes
    clss      = unique(labs(:)); 
    clss(1)   = [];
    Nc        = length(clss);

    tran      = [];
    test      = [];
    for nc = 1:Nc
        idx         = find(labs(:) == clss(nc));
        idx         = idx(randperm(length(idx)));
       
        if length(idx) < Nt
            tran        = [tran; idx(1:5)];
            test        = [test; idx(5+1:end)];
        else
            tran        = [tran; idx(1:Nt)];
            test        = [test; idx(Nt+1:end)];
        end
    end
    tran  = tran(:);
    test  = test(:);
end