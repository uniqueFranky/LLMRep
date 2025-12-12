import subprocess

if __name__ == '__main__':
    # Example usage of subprocess to run a Python module

    for use_penalty in [False, True]:
        for use_sae in [False, True]:
            for use_neuron in [False, True]:
                for decoder in ['greedy', 'top_k', 'top_p']:
                    cmd = [
                        'python', '-m', 'src.evaluation.hybrid_iwslt',
                        '--model', 'gpt2',
                        '--dataset', 'iwslt',
                        '--device', 'cuda:6',
                        '--max-samples', '1000',
                        '--output-file', f'hybrid_results/iwslt_{"penalty" if use_penalty else "no_penalty"}_{"sae" if use_sae else "no_sae"}_{"neuron" if use_neuron else "no_neuron"}_{decoder}.jsonl',
                        '--decode-strategy', decoder,
                        '--max-length', '150'
                    ]
                    if use_penalty:
                        cmd.append('--use-penalty')
                    if use_sae:
                        cmd.append('--use-sae')
                    if use_neuron:
                        cmd.append('--use-neuron')

                    print(f'Running command: {" ".join(cmd)}')
                    subprocess.run(cmd)
                    
