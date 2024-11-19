function [sN] = ReducedBaseOnline(datapoint, bb)
    % Reduced matrix A
    A = zeros(bb.N, bb.N);
    for q = 1:bb.Q
        A = A + fin(datapoint, q) * squeeze(bb.ANq(q, :, :));
    end

    % Reduced solution uN
    uN = A \ bb.FN;

    % Compute sN
    sN = dot(uN, bb.FN);
end

function [Fq] = fin(datapoint, q)
    Fq = datapoint(q);
end
