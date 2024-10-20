import matplotlib.pyplot as plt
import numpy as np

handKeypoints = {
    "x": [
        -0.010474077425897121, 0.11360502243041992, 0.3561234772205353,
        0.15308474004268646, -0.1063535138964653, 0.33356356620788574,
        0.294083833694458, 0.22640426456928253, 0.23204419016838074,
        0.11924495548009872, -0.01611400954425335, -0.021754136309027672,
        -0.01611400954425335, -0.07815365493297577, -0.15711310505867004,
        -0.16275303065776825, -0.15147316455841064, -0.25863248109817505,
        -0.3037521243095398, -0.2924722731113434, -0.25299254059791565
    ],
    "y": [
        -0.26991233229637146, -0.3037521243095398, -0.21915274858474731,
        -0.10071348398923874, -0.13455326855182648, 0.27716395258903503,
        0.20384442806243896, 0.057205405086278915, -0.01611410640180111,
        0.32792362570762634, 0.05156547203660011, -0.01611410640180111,
        -0.004834144376218319, 0.294083833694458, -0.01611410640180111,
        -0.0612337626516819, -0.08379358798265457, 0.2320442944765091,
        -0.004834144376218319, -0.08379358798265457, -0.12891334295272827
    ]
}

data = [
    handKeypoints[:, 0:5],
    np.insert(handKeypoints[:, 5:9].T, 0, handKeypoints[:, 0], axis=0).T,
    np.insert(handKeypoints[:, 9:13].T, 0, handKeypoints[:, 0], axis=0).T,
    np.insert(handKeypoints[:, 13:17].T, 0, handKeypoints[:, 0], axis=0).T,
    np.insert(handKeypoints[:, 17:21].T, 0, handKeypoints[:, 0], axis=0).T,
]

# Modify your data to fit the code
modified_data = []  # Create an empty list to store the modified data
for d in data:
    modified_d = np.array([d[:, 0], d[:, 1]])  # Modify the data format as per the code
    modified_data.append(modified_d)
canvas.draw()