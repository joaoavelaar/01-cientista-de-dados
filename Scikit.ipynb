{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOoqEGyqwT3g3UBb3J+vblX",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/joaoavelaar/machine-learning/blob/main/Scikit.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "PXnqRsmwcxP_"
      },
      "outputs": [],
      "source": [
        "# João Avelar\n",
        "# Importando Matplotlib e Numpy\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "%matplotlib inline"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Diâmetros (cm)\n",
        "Diametros = [[7], [10], [15], [30], [45]]\n",
        "\n",
        "# Preços (R$)\n",
        "Precos = [[10], [15], [18], [38.5], [52]]"
      ],
      "metadata": {
        "id": "UZh0HloUcytR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure()\n",
        "plt.xlabel('Diâmetro(cm)')\n",
        "plt.ylabel('Preço(R$)')\n",
        "plt.title('Diâmetro x Preço')\n",
        "plt.plot(Diametros, Precos, 'k.')\n",
        "plt.axis([0, 60, 0, 60])\n",
        "plt.grid(True)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 472
        },
        "id": "kMe9j9dfc4zZ",
        "outputId": "74585820-4581-413a-9566-2803a2cabaa7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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AMDWCDsAAMDWCDsAAMDW3Bp2Ro8eLQ8PD5efBg0aONfn5OQoLi5OAQEBqly5snr06KGMjAw3VgwAAMoat+/ZadSokY4dO+b8+eKLL5zrhg8frhUrVmjx4sVKTU3V0aNH1b17dzdWCwAAyhq3X3pevnx5BQUFFRjPzMxUYmKiFixYoJiYGEnS7Nmz1bBhQ23cuFG33377710qAAAog9wedvbt26datWrJ19dX0dHRGjt2rEJDQ7Vt2zbl5eWpY8eOzm0bNGig0NBQbdiw4ZphJzc3V7m5uc7lrKwsSVJeXp7y8vJu7IexuSv9o48lQx+tQy+tQy+tQR+tY2UP3Rp22rRpozlz5qh+/fo6duyYxowZoz/96U/avXu30tPT5e3trWrVqrm8JjAwUOnp6decc+zYsRozZkyB8ZSUFFWsWNHqj/CHlJSU5O4SbIE+WodeWodeWoM+ltyFCxcsm8utYefee+91/rlJkyZq06aNwsLC9N///lcVKlQo1pwjRoxQfHy8czkrK0u1a9dW+/btFRAQUOKa/8jy8vKUlJSkTp06ycvLy93llFn00Tr00jr00hr00TqnTp2ybC63H8b6pWrVqqlevXrav3+/OnXqpIsXL+rs2bMue3cyMjKueo7PFT4+PvLx8Skw7uXlxS+eReilNeijdeildeilNehjyVnZP7dfjfVL586d04EDBxQcHKyWLVvKy8tLycnJzvV79+7VkSNHFB0d7cYqAQBAWeLWPTvPPPOMunTporCwMB09elSjRo2Sp6en/vrXv8rPz0/9+/dXfHy8/P39VbVqVQ0ePFjR0dFciQUAAArNrWEnLS1Nf/3rX3Xq1CnVqFFDd955pzZu3KgaNWpIkiZOnKhy5cqpR48eys3NVWxsrN5++213lgwAAMoYt4adRYsW/eZ6X19fJSQkKCEh4XeqCAAA2E2pOmcHAADAaoQdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga4QdAABga6Um7IwbN04eHh4aNmyYcywnJ0dxcXEKCAhQ5cqV1aNHD2VkZLivSAAAUOaUirCzZcsWzZgxQ02aNHEZHz58uFasWKHFixcrNTVVR48eVffu3d1UJQAAKIvKu7uAc+fOqU+fPnrnnXf0yiuvOMczMzOVmJioBQsWKCYmRpI0e/ZsNWzYUBs3btTtt99+1flyc3OVm5vrXM7KypIk5eXlKS8v7wZ+Evu70j/6WDL00Tr00jr00hr00TpW9tDDGGMsm60Y+vXrJ39/f02cOFHt2rVTs2bNNGnSJK1du1YdOnTQmTNnVK1aNef2YWFhGjZsmIYPH37V+UaPHq0xY8YUGF+wYIEqVqx4oz4GAACw0IULF/S3v/1NmZmZqlq1aonmcuuenUWLFmn79u3asmVLgXXp6eny9vZ2CTqSFBgYqPT09GvOOWLECMXHxzuXs7KyVLt2bbVv314BAQGW1f5HlJeXp6SkJHXq1EleXl7uLqfMoo/WoZfWoZfWoI/WOXXqlGVzuS3s/Pjjjxo6dKiSkpLk6+tr2bw+Pj7y8fEpMO7l5cUvnkXopTXoo3XopXXopTXoY8lZ2T+3naC8bds2HT9+XC1atFD58uVVvnx5paamasqUKSpfvrwCAwN18eJFnT171uV1GRkZCgoKck/RAACgzHHbnp0OHTpo165dLmOPPvqoGjRooGeffVa1a9eWl5eXkpOT1aNHD0nS3r17deTIEUVHR7ujZAAAUAa5LexUqVJFt956q8tYpUqVFBAQ4Bzv37+/4uPj5e/vr6pVq2rw4MGKjo6+5pVYAAAAv+b2S89/y8SJE1WuXDn16NFDubm5io2N1dtvv+3usgAAQBlSqsLOunXrXJZ9fX2VkJCghIQE9xQEoFRLS0vTrl271KRJE0VERLi7HAClVKm4gzIAFFViYqIiIyM1cuRIRUZGKjEx0d0lASilCDsAypy0tDQNGDBADodDkuRwODRw4EClpaW5uTIApRFhB0CZs2/fPmfQuSI/P1/79+93U0UASjPCDoAyp27duipXzvU/X56enoqMjHRTRQBKM8IOgDInJCREM2fOlKenp6TLQWfGjBkKCQlxc2UASqNSdTUWABRW//79FRMTo/nz56tPnz5cjQXgmtizA6DMCgkJUePGjdmjA+A3EXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtEXYAAICtlS/Oi3Jzc7Vp0yYdPnxYFy5cUI0aNdS8eXNFRERYXR8AAECJFCnsrF+/XpMnT9aKFSuUl5cnPz8/VahQQadPn1Zubq7q1KmjAQMG6IknnlCVKlVuVM0AAACFVujDWF27dlWvXr0UHh6uTz/9VNnZ2Tp16pTS0tJ04cIF7du3Ty+++KKSk5NVr149JSUl3ci6AQAACqXQe3buv/9+LVmyRF5eXlddX6dOHdWpU0f9+vXTnj17dOzYMcuKBAAAKK5Ch52BAwcWetKoqChFRUUVqyAAAAArWXY1Vn5+vlVTAQAAWKbIYefzzz93OR/n+PHjuvPOO+Xj46OWLVtq3759lhYIAABQEkUOOy+99JIOHTrkXB41apQuXryoZcuWqVatWho6dKiV9QEAAJRIke+zs3fvXrVs2dK5vHz5cs2fP1/t2rVT06ZN1bRpU0sLBAAAKIlCh51HH31UknT69Gm99tprqlKlik6dOqWTJ0/qvffe03vvvSeHw6Hs7Gw99thjkqRZs2bdmKoBAAAKqdBhZ/bs2ZKkrVu3qkOHDnryySc1YcIEZWZmOkNNWlqa1qxZQ8gBAAClRpEPYz355JMaPHiwJk2apB9++EEffvihc92nn36q2267zdICAQAASqLIYeepp55SgwYNtGPHDkVHR+uOO+5wrvP29taIESMsLRAAAKAkivUg0JiYGMXExBQYf/jhh0tcEAAAgJUsu6mgJG3fvl2dO3e2ckoAAIASKXLYWbNmjZ555hk9//zz+uGHHyRJ3333nbp166bWrVvL4XBYXiQAAEBxFekwVmJioh5//HH5+/vrzJkz+s9//qM333xTgwcPVq9evbR79241bNjwRtUKAABQZEXaszN58mSNHz9eJ0+e1H//+1+dPHlSb7/9tnbt2qXp06cTdAAAQKlTpLBz4MAB9ezZU5LUvXt3lS9fXq+//rpCQkJuSHEAAAAlVaSw8/PPP6tixYqSJA8PD/n4+Cg4OPiGFAYAAGCFIl96/p///EeVK1eWJF26dElz5sxR9erVXbYZMmSINdUBAACUUJHCTmhoqN555x3nclBQkObOneuyjYeHB2EHAACUGkUKO4cOHbpBZQAAANwYlt5UEAAAoLQpdNhZtGhRoSf98ccftX79+mIVBAAAYKVCh51p06apYcOGmjBhgr799tsC6zMzM7Vy5Ur97W9/U4sWLXTq1ClLCwUAACiOQoed1NRUjR8/XklJSbr11ltVtWpV1a1bV40bN1ZISIgCAgL02GOPKTQ0VLt371bXrl2vO+e0adPUpEkTVa1aVVWrVlV0dLRWrVrlXJ+Tk6O4uDgFBASocuXK6tGjhzIyMor3SQEAwB9SkU5Q7tq1q7p27aqTJ0/qiy++0OHDh/Xzzz+revXqat68uZo3b65y5Qp/GlBISIjGjRununXryhijd999Vw888IB27NihRo0aafjw4frkk0+0ePFi+fn5adCgQerevTuHyAAAQKEV+T47klS9enV169atxG/epUsXl+VXX31V06ZN08aNGxUSEqLExEQtWLBAMTExkqTZs2erYcOG2rhxo26//fYSvz8AALC/YoWdK7Zt2+Y8fycqKkotWrQo9lz5+flavHixzp8/r+joaG3btk15eXnq2LGjc5sGDRooNDRUGzZsuGbYyc3NVW5urnM5KytLkpSXl6e8vLxi1wc5+0cfS4Y+WodeWodeWoM+WsfKHhYr7Bw/fly9e/fWunXrVK1aNUnS2bNn1b59ey1atEg1atQo9Fy7du1SdHS0cnJyVLlyZS1dulRRUVHauXOnvL29nfNfERgYqPT09GvON3bsWI0ZM6bAeEpKivNRFyiZpKQkd5dgC/TROvTSOvTSGvSx5C5cuGDZXMUKO4MHD1Z2dra++eYb55PO9+zZo379+mnIkCFauHBhoeeqX7++du7cqczMTH3wwQfq16+fUlNTi1OWJGnEiBGKj493LmdlZal27dpq3769AgICij0vLqfspKQkderUSV5eXu4up8yij9ahl9ahl9agj9ax8qruYoWd1atX67PPPnMGHenyYayEhATdfffdRZrL29tbkZGRkqSWLVtqy5Ytmjx5snr16qWLFy/q7NmzLnt3MjIyFBQUdM35fHx85OPjU2Dcy8uLXzyL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0pSSkqK0tDR3lwIAQKlS3p1vfu+99+ree++96jpjjCZNmqQXX3xRDzzwgCTpvffeU2BgoJYtW6bevXv/nqWWaomJiRowYIAcDofKlSunmTNnqn///u4uCwCAUsGtYee3HDx4UOnp6erYsaNzzM/PT23atNGGDRuuGXZyc3OVm5vrXM7KypIk5eXlKS8v78YW7QZpaWnOoCNJDodDAwcOVExMjEJCQix9ryv9s2Mff0/00Tr00jr00hr00TpW9rDUhp309HRJUmBgoMt4YGCgc93VjB07VmPGjCkwnpKSoooVK1pbZCmwa9cuZ9C5Ij8/X/Pnz1fjxo1vyHsmJSXdkHn/aOijdeildeilNehjyV24cMGyuUpt2CmuESNGKD4+3rmclZWl2rVrq3379goICHBjZTdGkyZNNGrUKJfA4+npqT59+tyQPTtJSUnq1KmTvLy8LJ37j4Q+WodeWodeWoM+WufUqVOWzVVqw05QUJAkKSMjQ8HBwc7xjIwMNWvW7Jqv8/HxkY+PT4FxLy8vW/7iRUREaObMmRo4cKDy8/Pl6empGTNmKCIi4oa9p117+Xujj9ahl9ahl9agjyVnZf9K7X12IiIiFBQUpOTkZOdYVlaWNm3apOjoaDdWVvr0799fhw4dUkpKig4dOsTJyQAA/IJb9+ycO3dO+/fvdy4fPHhQO3fulL+/v0JDQzVs2DC98sorqlu3riIiIjRy5EjVqlVL3bp1c1/RpVRISIjlh60AALADt4adrVu3qn379s7lK+fa9OvXT3PmzNE///lPnT9/XgMGDNDZs2d15513avXq1fL19XVXyQAAoIxxa9hp166djDHXXO/h4aGXX35ZL7/88u9YFQAAsJNSe84OAACAFQg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA1gg7AADA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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Importando o módulo de Regressão Linear do scikit-learn\n",
        "from sklearn.linear_model import LinearRegression"
      ],
      "metadata": {
        "id": "ErWXe-dVc7E6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Preparando os dados de treino\n",
        "\n",
        "# Vamos chamar de X os dados de diâmetro da Pizza.\n",
        "X = [[7], [10], [15], [30], [45]]\n",
        "\n",
        "# Vamos chamar de Y os dados de preço da Pizza.\n",
        "Y = [[10], [15], [18], [38.5], [52]]"
      ],
      "metadata": {
        "id": "XSHt_3ZXdBf7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Criando o modelo\n",
        "modelo = LinearRegression()"
      ],
      "metadata": {
        "id": "x2A7dKjLdGVi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "type(modelo)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YPjygJ8YdImQ",
        "outputId": "2bc34268-a065-4e2f-d8ec-484a96cefa35"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "sklearn.linear_model._base.LinearRegression"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Treinando o modelo\n",
        "modelo.fit(X, Y)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "id": "TZV5vbnjdJ6U",
        "outputId": "bfbf568d-5887-4665-924a-9a6f725b1483"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LinearRegression()"
            ],
            "text/html": [
              "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Prevendo o preço de uma pizza de 20 cm de diâmetro\n",
        "print(\"Uma pizza de 20 cm de diâmetro deve custar: R$ %.2f\" % modelo.predict([20][0]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 484
        },
        "id": "OCWW3R3odLde",
        "outputId": "4e26fa04-7f26-4dcd-90d0-1cb8ae49037b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "error",
          "ename": "ValueError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-18-0d8d728a46f2>\u001b[0m in \u001b[0;36m<cell line: 2>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# Prevendo o preço de uma pizza de 20 cm de diâmetro\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Uma pizza de 20 cm de diâmetro deve custar: R$ %.2f\"\u001b[0m \u001b[0;34m%\u001b[0m \u001b[0mmodelo\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m20\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_base.py\u001b[0m in \u001b[0;36mpredict\u001b[0;34m(self, X)\u001b[0m\n\u001b[1;32m    352\u001b[0m             \u001b[0mReturns\u001b[0m \u001b[0mpredicted\u001b[0m \u001b[0mvalues\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    353\u001b[0m         \"\"\"\n\u001b[0;32m--> 354\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_decision_function\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    355\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    356\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_set_intercept\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mX_offset\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_offset\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mX_scale\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_base.py\u001b[0m in \u001b[0;36m_decision_function\u001b[0;34m(self, X)\u001b[0m\n\u001b[1;32m    335\u001b[0m         \u001b[0mcheck_is_fitted\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    336\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 337\u001b[0;31m         \u001b[0mX\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_validate_data\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maccept_sparse\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"csr\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"csc\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"coo\"\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mreset\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    338\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0msafe_sparse_dot\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcoef_\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mT\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdense_output\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m+\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mintercept_\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    339\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/sklearn/base.py\u001b[0m in \u001b[0;36m_validate_data\u001b[0;34m(self, X, y, reset, validate_separately, **check_params)\u001b[0m\n\u001b[1;32m    563\u001b[0m             \u001b[0;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Validation should be done on X, y or both.\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    564\u001b[0m         \u001b[0;32melif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mno_val_X\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0mno_val_y\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 565\u001b[0;31m             \u001b[0mX\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcheck_array\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minput_name\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m\"X\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mcheck_params\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    566\u001b[0m             \u001b[0mout\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mX\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    567\u001b[0m         \u001b[0;32melif\u001b[0m \u001b[0mno_val_X\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mno_val_y\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36mcheck_array\u001b[0;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)\u001b[0m\n\u001b[1;32m    892\u001b[0m             \u001b[0;31m# If input is scalar raise error\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    893\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0marray\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mndim\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 894\u001b[0;31m                 raise ValueError(\n\u001b[0m\u001b[1;32m    895\u001b[0m                     \u001b[0;34m\"Expected 2D array, got scalar array instead:\\narray={}.\\n\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    896\u001b[0m                     \u001b[0;34m\"Reshape your data either using array.reshape(-1, 1) if \"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mValueError\u001b[0m: Expected 2D array, got scalar array instead:\narray=20.\nReshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Coeficientes\n",
        "print('Coeficiente: \\n', modelo.coef_)\n",
        "\n",
        "# MSE (mean square error)\n",
        "print(\"MSE: %.2f\" % np.mean((modelo.predict(X) - Y) ** 2))\n",
        "\n",
        "# Score de variação: 1 representa predição perfeita\n",
        "print('Score de variação: %.2f' % modelo.score(X, Y))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8GVHoIUbdNcX",
        "outputId": "be3f7010-6516-4af9-825c-2e1afabb0d6b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Coeficiente: \n",
            " [[1.11781609]]\n",
            "MSE: 1.96\n",
            "Score de variação: 0.99\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Scatter Plot representando a regressão linear\n",
        "plt.scatter(X, Y,  color = 'black')\n",
        "plt.plot(X, modelo.predict(X), color = 'green', linewidth = 3)\n",
        "plt.xlabel('X')\n",
        "plt.ylabel('Y')\n",
        "plt.xticks(())\n",
        "plt.yticks(())\n",
        "\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 426
        },
        "id": "z6s-iodFeNlb",
        "outputId": "8a15aa38-4710-4199-e6b7-2f6090ba74e4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "Nxqth_T4eP1F"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "LDnv9jixekIz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "Ft4N633Pelq3"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
