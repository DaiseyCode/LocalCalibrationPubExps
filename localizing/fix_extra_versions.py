from lmwrapper.openai_wrapper import OpenAiModelNames, get_open_ai_lm
from synthegrator.execution_threading import solve_and_evaluate_problems

from fixsolver import RewriteFixSolver, solution_to_repair_problem, render_rewrite_prompt, parse_rewrite
from localizing.multi_data_gathering import get_base_solves
from synthegrator.synthdatasets import DatasetName

def main():
    locs = get_base_solves(
        use_eval_plus=True,
        dataset=DatasetName.humaneval,
        gen_model_name=OpenAiModelNames.gpt_4o_mini,
        filter_to_original_fails=True,
    )
    print(len(list(locs.iter_passed_filtered())))
    lm = get_open_ai_lm(OpenAiModelNames.o3_mini)
    solver = RewriteFixSolver(lm)
    num_fixed = 0
    for loc in locs.iter_passed_filtered():
        print("Base text")
        print(loc.get_base_text())
        #print(loc.base_eval.test_results)
        #for test in loc.base_eval.test_results:
        #    print(test.fail_text)
        new_prob = solution_to_repair_problem(loc.base_eval)
        prompt = render_rewrite_prompt(new_prob, 1.0, True)
        pred = lm.predict(prompt)
        print('completion text')
        print(pred.completion_text)
        print("parse")
        solve_steps = parse_rewrite(pred, new_prob)
        print(solve_steps[0])
        print("Now run full solver")
        sol = solver.solve(new_prob)
        print("solve result vlaue")
        print(sol.solve_steps[0].value)
        print("do eval")
        new_eval = sol.problem.preferred_solution_evaluator.evaluate(sol)
        print("New metric", new_eval.main_metric.is_success)
        if new_eval.main_metric.is_success:
            num_fixed += 1
    print("num_fixed", num_fixed)


if __name__ == "__main__":
    main()