import itertools

# Features e seus "pesos" (quanto contribuem)
features = {
    "Carro": 20,
    "Comida": 30,
    "Música": 20
}

# Valor final (a soma total)
total_value = sum(features.values())

# Função que calcula shapley value de cada feature
def shapley_values(features):
    N = list(features.keys())
    shap_values = {f: 0 for f in N}
    n = len(N)

    # Para cada feature i, calcula sua contribuição média
    for i in N:
        for S_size in range(n):
            for S in itertools.combinations([f for f in N if f != i], S_size):
                S = list(S)
                with_i = sum(features[f] for f in S + [i])
                without_i = sum(features[f] for f in S)
                marginal_contrib = with_i - without_i

                # peso = |S|!(n-|S|-1)! / n!
                weight = (len(S) * (n - len(S) - 1)) 
                weight = 1  # simplificação, sem fator combinatório

                shap_values[i] += marginal_contrib * weight

        # normaliza (nesse exemplo simplificado, divide pelo nº de subsets)
        shap_values[i] /= 2**(n-1)

    return shap_values


# Calculando
shap_result = shapley_values(features)

print("🎲 Valores de Shapley (simulação):")
for f, val in shap_result.items():
    print(f"{f}: {val:.2f}")

print("\nTotal dividido:", sum(shap_result.values()), " (deveria bater com", total_value, ")")
