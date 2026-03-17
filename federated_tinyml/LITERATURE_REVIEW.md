# Literature Review

## Paper identity
- Full title: Federated Learning framework for LoRaWAN-enabled IIoT communication: A case study
- Authors: Oscar Torres Sanchez, Guilherme Borges, Duarte Raposo, Andre Rodrigues, Fernando Boavida, Jorge Sa Silva (University of Coimbra, Portugal)
- Published: arXiv:2410.11612v1, October 2024


## Experimental setup
- Devices: 4 IIoT prototype devices monitoring real civil construction machinery (Manitou multifunctions, Atlas Copco drill, Jaw Crusher, Doosan wheel loader) at a real construction site in Vilar Formoso, Portugal.
- Network path: Private LoRaWAN -> TTN -> MQTT -> data management agent.
- Data collected: 77,798 total messages; features include battery voltage, fuel consumption, RPM, water/oil temperature, oil pressure.
- FL framework: Flower (Python FL orchestration), emulating 4 virtual clients (one per machine) with one central server.
- ML model: Compact autoencoder (encoder + decoder), one hidden layer of 32 neurons, Tanh activation, Adam optimizer, MSE loss, designed to remain small for LoRaWAN transmission.

---



## Key quantitative results

### FL vs centralized performance

| Model | Mean Accuracy | Mean F1-Score | TNR (Specificity) |
|---|---:|---:|---:|
| OC-SVM (centralized) | 92.06% | 94.59% | 81.49% |
| Isolation Forest (centralized) | 93.46% | 95.57% | 82.44% |
| Autoencoder (centralized) | 93.13% | 95.36% | 80.82% |
| Autoencoder FL (federated) | 92.30% | 94.77% | 90.65% |

Interpretation from the review notes:
- FL reaches comparable accuracy to centralized approaches.
- FL yields higher specificity (90.65%), implying fewer false alarms.

---

## LoRaWAN communication analysis (most relevant part)

### Model size vs transmission burden
- Optimized AE (32 hidden neurons): 1.39 KB of model weights per FL round.
- With 128 neurons: 5.52 KB.
- With 3 hidden layers: up to 44 KB.

### Message-count formula

Nm = ceil(Msize / MaxPayload) * Rd

Where:
- Nm = number of LoRaWAN messages
- Msize = model-update size
- MaxPayload depends on spreading factor (SF)
- Rd = number of aggregation rounds

Payload assumptions in the notes:
- SF7/SF8 = 222 bytes
- SF9/SF10 = 115 bytes
- SF11/SF12 = 51 bytes

### Example in the notes
For the 1.39 KB model:
- Best-case range: 7 messages (SF7, 1 round)
- Worst-case range: 2,233 messages (SF12, 80 rounds)

### Additional communication findings
- Total airtime for training messages: 52.8 minutes (optimal configuration in the cited setup).
- Best round/epoch setting reported: 20 epochs x 4 rounds
  - F1 = 95.23%
  - Accuracy = 92.94%
  - TNR = 93%
- This outperformed extreme schedules such as 80 rounds x 1 epoch or 1 round x 80 epochs.

---

## Centralized vs federated differences (summary table)

| Aspect | Centralized | Federated (paper setup) |
|---|---|---|
| Data movement | Raw data uploaded to server | Raw data remains local; model updates sent |
| Privacy | Lower (data leaves device) | Higher (weights/updates shared) |
| Network load | Heavy data transport | Model-update transport, still constrained by SF/payload |
| Accuracy | Slightly higher in some models | Comparable overall |
| Specificity (false alarms) | Lower in compared centralized AEs | Higher in AE-FL (90.65%) |
| Optimization target | Pure model accuracy | Accuracy + communication feasibility |

---

## Main conclusions captured from the review notes
1. FL over LoRaWAN is feasible, even with non-IID and unbalanced client data.
2. Round/epoch design strongly affects both model quality and communication cost.
3. Fewer rounds with more local epochs can reduce message burden substantially.
4. Model size must be minimized for LoRaWAN viability.
5. The paper explicitly motivates future work on TinyML methods (quantization, pruning, compression) for constrained microcontrollers.

---

## Relevance to our thesis direction
This reviewed paper supports the methodological foundation for:
- FL orchestration under LoRaWAN constraints
- message-budget and airtime-aware FL design
- selecting practical round/epoch schedules
- shrinking model size for real edge deployment

It is directly useful as a protocol-and-systems reference while our thesis application focus remains:
Decentralized Edge Intelligence for LoRaWAN: Federated Learning for Environment-Driven Path Loss and Link Quality Modeling.


![](https://github.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/blob/83aef0bfd7c25b2ed670212028a70b6d7b3cee25/pics/1.1.png)


![](https://github.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/blob/83aef0bfd7c25b2ed670212028a70b6d7b3cee25/pics/1.2.png)

![](https://github.com/pratik001010/Decentralized-Edge-Intelligence-for-LoRaWAN-Federated-Learning-for-Environment-Driven-Path-Loss-/blob/83aef0bfd7c25b2ed670212028a70b6d7b3cee25/pics/1.3.png)

