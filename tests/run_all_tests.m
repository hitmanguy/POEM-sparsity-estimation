function all_passed = run_all_tests()
% RUN_ALL_TESTS Execute all unit tests for the POEM sparsity estimator.
%
% Inputs:
%   None
%
% Outputs:
%   all_passed - Logical true if all tests pass (1x1 logical)

    root_dir = fileparts(fileparts(mfilename('fullpath')));
    run(fullfile(root_dir, 'startup_poem.m'));

    fprintf('==============================================================\n');
    fprintf('Running POEM Test Suite (MATLAB %s)\n', version);
    fprintf('==============================================================\n');

    p1 = test_moment_inversion_exact();
    p2 = test_bracket_contains_k();

    all_passed = p1 && p2;
    fprintf('==============================================================\n');
    if all_passed
        fprintf('ALL TESTS PASSED.\n');
    else
        error('One or more tests failed.');
    end
end
