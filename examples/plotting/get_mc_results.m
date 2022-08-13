function model_info = get_mc_results(model_info, problem_path, fp_tol)
    results_dir = dir([problem_path, 'results/monte_carlo/']);

    model_info.median_final_time = nan*ones(height(model_info),1);
    model_info.max_final_time = nan*ones(height(model_info),1);

    model_info.frac_stable = nan*ones(height(model_info),1);

    model_info.median_final_dist = nan*ones(height(model_info),1);
    model_info.max_final_dist = nan*ones(height(model_info),1);

    model_info.median_subopt = nan*ones(height(model_info),1);
    model_info.max_subopt = nan*ones(height(model_info),1);

    % Go through test results directory and load results, look for optimal
    % final times and costs
    % Skip first two entries since these are just '.' and '..'
    for i=3:size(results_dir, 1)
        results_path = [problem_path, 'results/monte_carlo/', results_dir(i).name];
        results = load(results_path);

        if exist('optimal_results', 'var')
            for j=1:size(optimal_results.ocp_converged, 2)
                if results.ocp_converged(j)
                    if or(~optimal_results.ocp_converged(j),...
                            results.opt_final_times(j) < optimal_results.opt_final_times(j))
                        optimal_results.opt_final_times(j) = results.opt_final_times(j);
                    end
                    if or(~optimal_results.ocp_converged(j),...
                            results.opt_costs(j) < optimal_results.opt_costs(j))
                        optimal_results.opt_costs(j) = results.opt_costs(j);
                    end
                    optimal_results.ocp_converged(j) = 1;
                end
            end
        else
            optimal_results = results;
        end
    end

    % Go through test results directory and load results
    % Skip first two entries since these are just '.' and '..'
    for i=3:size(results_dir, 1)
        timestamp = results_dir(i).name(6:end-4);
        results_path = [problem_path, 'results/monte_carlo/', results_dir(i).name];
        results = load(results_path);

        % Which row of model info does this timestamp correspond to?
        j = find(ismember(model_info.timestamp, str2num(timestamp)));

        %model_info.median_final_time(j) = median(results.NN_final_times);
        %model_info.max_final_time(j) = max(results.NN_final_times);
        
        final_times = results.NN_final_times ./ optimal_results.opt_final_times;
        model_info.median_final_time(j) = median(final_times);
        model_info.max_final_time(j) = max(final_times);

        model_info.frac_stable(j) = mean(results.final_dists <= fp_tol);

        model_info.median_final_dist(j) = median(results.final_dists);
        model_info.max_final_dist(j) = max(results.final_dists);

        subopt = 100 * (results.NN_costs - optimal_results.opt_costs) ./ optimal_results.opt_costs;
        subopt(find(~optimal_results.ocp_converged)) = 0;
        subopt = max(subopt, 1e-07);

        model_info.median_subopt(j) = median(subopt);
        model_info.max_subopt(j) = max(subopt);
    end
end

