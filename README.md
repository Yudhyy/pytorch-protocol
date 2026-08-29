# pytorch protocol

i use this repo to learn pytorch. i will build neural nets from scratch, project stuff, and take notes.

## what is here

- `main.py`: my marimo notebook. i write code and notes here.
- `books`: pdfs and reading material.
- `data`: images and datasets for tests.
- `public`: charts and diagrams.
- `layouts`: how i arrange the marimo grid and slides.

## how to run

1. install marimo:
   ```bash
   uv sync
   ```
2. run the notebook:
   ```bash
   marimo edit main.py
   ```
3. open your browser to the link it shows.

## roadmap

- [x] essential pytorch: tensors, memory, autograd, and operations.
- [x] learning a line from scratch (manually update): forward pass, mse, backprop, and gradient descent.
- [ ] pytorch idiomatic training: torch.nn, torch.optim, and the professional loop.
- [ ] the transformer's feed-forward network.