import subprocess

if __name__ == '__main__':
    # Example usage of subprocess to run a Python module

    for model in ['gpt2', 'google/gemma-2-2b']:
        for use_penalty in [False, True]:
            for use_sae in [False, True]:
                for use_neuron in [False, True]:
                    for decoder in ['greedy', 'top_k', 'top_p']:
                        if use_sae and use_neuron:
                            continue  # skip invalid combination

                        if use_sae:
                            model_script = 'src.evaluation.sae_decode_penalty'
                            cmd = [
                                'python', '-m', model_script,
                                '--model', model,
                                '--device', 'cuda:4',
                                '--decode-strategy', decoder,
                                '--max-samples', '1000',
                                '--output-file', f'separate_hybrid_results/{model}_sae_{"penalty" if use_penalty else "nopenalty"}_{decoder}.jsonl',
                            ]
                            if use_penalty:
                                cmd.append('--use-penalty')
                        elif use_neuron:
                            model_script = 'src.evaluation.neuron_penalty'
                            cmd = [
                                'python', '-m', model_script,
                                '--model', model,
                                '--device', 'cuda:4',
                                '--decode-strategy', decoder,
                                '--max-samples', '1000',
                                '--output-file', f'separate_hybrid_results/{model}_neuron_{"penalty" if use_penalty else "nopenalty"}_{decoder}.jsonl',
                            ]
                            if use_penalty:
                                cmd.append('--use-penalty')
                        else:
                            continue  # skip if neither SAE nor Neuron penalty is used
                        # Run the command
                        print(f"Running command: {' '.join(cmd)}")
                        subprocess.run(cmd, check=True)
                    
