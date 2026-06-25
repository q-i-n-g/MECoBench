# Full Task Files

Place the full MECoBench task files here after downloading them from
[Hugging Face](https://huggingface.co/datasets/q-i-n-g/MECoBench):

- `parallel.json`
- `sequential.json`

The GitHub repository includes five-case examples under `data/examples/`.

Example:

```bash
mkdir -p data/task
huggingface-cli download q-i-n-g/MECoBench task/parallel.json --repo-type dataset --local-dir .
huggingface-cli download q-i-n-g/MECoBench task/sequential.json --repo-type dataset --local-dir .
mv task/parallel.json data/task/parallel.json
mv task/sequential.json data/task/sequential.json
rmdir task
```
