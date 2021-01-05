function out = perform_thresholding(int, tau, type)

switch type
    case 'SOFT'
        out  = int.*max(0, 1-tau./max(abs(int), 1e-10));
    case 'HARD'
        out  = int.*(abs(int)>tau);
    otherwise
        disp('unknown')
end

end

