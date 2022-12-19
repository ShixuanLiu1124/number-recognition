#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ROW 28
#define COL 28

// split string into tokens with char use malloc
char **split(char *str, char delim, int *num) {

    char **tokens = NULL;
    char *token = NULL;
    int i = 0;

    token = strtok(str, &delim);
    while (token != NULL) {
        tokens = (char **) realloc(tokens, sizeof(char *) * (i + 1));
        tokens[i] = (char *) malloc(sizeof(char) * (strlen(token) + 1));
        strcpy(tokens[i], token);
        token = strtok(NULL, &delim);
        i++;
    }
    *num = i;

    return tokens;
}

// matrix multiplication
void multiply(float B[ROW * COL], float C[ROW * COL][128], float D[128]) {

    memset(D, 0, sizeof(float) * 128);
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < ROW * COL; j++) {
            D[i] += B[j] * C[j][i];
        }
    }
}

void image_to_matrix(float A[ROW][COL], FILE *fp) {

    // Use the fgets function to skip the first four lines in the file
    char line[10000];
    for (int i = 0; i < 4; ++i) {
        fgets(line, 10000, fp);
    }

    // Use fgetc function repeatedly to read each pixel from the fifth line until 784 chars have been obtained.
    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < COL; ++j) {
            A[i][j] = fgetc(fp) / 255.0;
        }
    }
}

void reshape(float A[ROW][COL], float B[ROW * COL]) {

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            B[i * 28 + j] = A[i][j];
        }
    }
}

void add_D_and_E_to_F(float D[128], float E[128], float F[128]) {

    for (int i = 0; i < 128; ++i) {
        F[i] = D[i] + E[i];
    }
}

void activate_F_to_G(float F[128], float G[128]) {

    // relu function
    for (int i = 0; i < 128; ++i) {
        if (F[i] < 0) {
            G[i] = 0;
        } else {
            G[i] = F[i];
        }
    }
}

void multiply_G_and_W2(float G[128], float W2[128][10], float H[10]) {

    memset(H, 0, sizeof(float) * 10);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 128; j++) {
            H[i] += G[j] * W2[j][i];
        }
    }
}

void add_H_and_B2_to_L(float H[10], float B2[10], float L[10]) {

    for (int i = 0; i < 10; ++i) {
        L[i] = H[i] + B2[i];
    }
}

void softmax_L_to_S(float L[10], float S[10]) {

    float sum = 0;
    for (int i = 0; i < 10; ++i) {
        sum += exp(L[i]);
    }
    for (int i = 0; i < 10; ++i) {
        S[i] = exp(L[i]) / sum;
    }
}

int max_index(float S[10]) {

    int max_index = 0;
    for (int i = 0; i < 10; ++i) {
        if (S[i] > S[max_index]) {
            max_index = i;
        }
    }

    return max_index;
}

int main(int argc, char *argv[]) {

    char y_or_n;
    char *input[1000 + 3];

    // main loop
    do {
        // input
        printf("please input filenames:");
        gets(input);

        int file_num = 0;
        char **filenames = NULL;

        // get filenames
        // if ',' exist
        if (strchr(input, ',') != NULL) {
            // split string into filenames with ','
            filenames = split(input, ',', &file_num);
        } else {
            file_num = 1;
            filenames = (char **) malloc(sizeof(char *));
            filenames[0] = (char *) malloc(sizeof(char) * (strlen(input) + 1));
            memcpy(filenames[0], input, strlen(input) + 1);
        }

        // number recognition
        for (int i = 0; i < file_num; ++i) {
            float A[28][28];
            FILE *fp = fopen(filenames[i], "rb");

            // invalid file
            if (fp == NULL) {
                puts("invalid file.");
                continue;
            }

            image_to_matrix(A, fp);
            fclose(fp);
            fp = NULL;

            float B[ROW * COL];
            reshape(A, B);

            float C[ROW * COL][128];
            fp = fopen("W1.txt", "r");
            // read matrix C from fp
            for (int row = 0; row < ROW * COL; ++row) {
                for (int col = 0; col < 128; ++col) {
                    fscanf(fp, "%f", &C[row][col]);
                }
            }
            float D[128];
            memset(D, 0, sizeof(float) * 128);
            multiply(B, C, D);
            fclose(fp);
            fp = NULL;

            float E[128];
            fp = fopen("B1.txt", "r");
            // read matrix E from fp
            for (int index = 0; index < 128; ++index) {
                fscanf(fp, "%f", &E[index]);
            }
            fclose(fp);
            fp = NULL;
            float F[128];
            add_D_and_E_to_F(D, E, F);

            float G[128];
            activate_F_to_G(F, G);

            fp = fopen("W2.txt", "r");
            float W2[128][10];
            // read matrix W2 from fp
            for (int row = 0; row < 128; ++row) {
                for (int col = 0; col < 10; ++col) {
                    fscanf(fp, "%f", &W2[row][col]);
                }
            }
            fclose(fp);
            fp = NULL;
            float H[10];
            memset(H, 0, sizeof(float) * 10);
            multiply_G_and_W2(G, W2, H);

            fp = fopen("B2.txt", "r");
            float B2[10];
            // read matrix B2 from fp
            for (int index = 0; index < 10; ++index) {
                fscanf(fp, "%f", &B2[index]);
            }
            fclose(fp);
            fp = NULL;
            float L[10];
            add_H_and_B2_to_L(H, B2, L);

            float S[10];
            softmax_L_to_S(L, S);

            // find the max index which is the max probability
            int res = max_index(S);

            // output result
            printf("%s:%d\n", filenames[i], res);
        }

        // memory free
        for (int i = 0; i < file_num; ++i) {

            // if filenames[i] not NULL,free it
            if (filenames[i] != NULL) {
                free(filenames[i]);
                filenames[i] = NULL;
            }
        }
        free(filenames);
        filenames = NULL;

        // continue or not
        printf("do you want to continue? please input [Y or N]:");
        scanf("%c", &y_or_n);
        getchar();
    } while (y_or_n == 'Y' || y_or_n == 'y');

    puts("Bye");

    return 0;
}
