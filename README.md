My re-implementation for https://paperswithcode.com/.../brain-tumor-segmentation...
Original implementation: https://github.com/naldeborgh7575/brain_segmentation
The paper introduces a CNN implementation for Cancer detection via BLAST Dataset (I used Blast2015), the original paper implemented a Two Path neural network (Using Graph NN but it's deprecated in keras, I will have to re-implement it via the functional API).
I have fixed some issues,  but as the author said it's outdated and he doesn't accept PR's anymore, so I created a separated repo, I have only implemented the CNN part, for the Graph part(Two path implementation), I'm still studying it, but I have fixed issues in Patching, also I think that I work on a different Dataset so I tuned the code to the data set I have.
*I don't have any rights for this implementation, I just fixed the issues in the original one.
