% =========================
% File: demo_Eikonal_3d_1st.m
% =========================
clear; clc;

% ===== Debug/Test switch =====
doTest = true;   % <-- 你要的“开始加一个判断，设为true用来测试”

if ~doTest
    disp('doTest=false, nothing to run. Set doTest=true to execute the demo.');
    return;
end

% ===== Grid setup (match your print: dx=0.0833 for n=25 on length 2) =====
n  = 25;
Lx = 2; Ly = 2; Lz = 2;
dx = Lx/(n-1); dy = Ly/(n-1); dz = Lz/(n-1);

% ===== Velocity model (positive everywhere) =====
c = ones(n,n,n);      % speed=1 everywhere (slowness=1)

% ===== 18 sources (copied from your logs) =====
src_list = [
    13  2   4
    24  13  4
    13  24  4
    13  13  4
     2   2  22
     2  13  4
     2  24  4
    24  24  4
    24   2  4
     2   2  4
    24  24  22
    24   2  22
     2  24  22
     2  13  22
    24  13  22
    13   2  22
    13  24  22
    13  13  22
];

fprintf('Number of sources: %d\n', size(src_list,1));

% ===== Solver options =====
opts = struct();
opts.max_sweeps   = 200;
opts.tol          = 1e-6;
opts.large        = 1e9;     % initial "infinity"
opts.debug        = true;    % print grid/source/iter lines
opts.debug_every  = 1;       % print every sweep
opts.warn_limit   = 12;      % avoid flooding warnings
opts.check_finite = true;    % warn if tau_new is nonfinite
opts.skip_source_update = true;

% ===== Run each source =====
for p = 1:size(src_list,1)
    src = src_list(p,:);
    fprintf('\n========== Starting Source %d/%d ==========\n', p, size(src_list,1));
    [tau, info] = Eikonal_3d_1st_(c, src, dx, dy, dz, opts); %#ok<NASGU>
end