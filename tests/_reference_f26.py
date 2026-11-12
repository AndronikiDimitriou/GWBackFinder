"""Literal transcription of the Julia ``f26`` (auto-generated from mock_signal.jl).

Used only as a test reference for the vectorised implementation."""


def f26_reference(S, S0, n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21, n22, n23, n24, n25, As, Sb1, Sb2, Sb3, Sb4, Sb5, Sb6, Sb7, Sb8, Sb9, Sb10, Sb11, Sb12, Sb13, Sb14, Sb15, Sb16, Sb17, Sb18, Sb19, Sb20, Sb21, Sb22, Sb23, Sb24, Sb25):
    if S <= Sb1:
        result = As * (Sb12 / S0)**(n12 - n11) * (Sb11 / S0)**(n11 - n10) * (Sb10 / S0)**(n10 - n9) * (Sb9 / S0)**(n9 - n8)*(Sb8 / S0)**(n8 - n7) * (Sb7 / S0)**(n7 - n6) * (Sb6 / S0)**(n6 - n5) * (Sb5 / S0)**(n5 - n4) * (Sb4 / S0)**(n4 - n3)*(Sb3 / S0)**(n3 - n2) * (Sb2 / S0)**(n2 - n1) * (Sb1 / S0)**(n1 - n0) * (S / S0)**(n0)
    elif Sb1 < S <= Sb2:
        result = As * (Sb12 / S0)**(n12 - n11) * (Sb11 / S0)**(n11 - n10) * (Sb10 / S0)**(n10 - n9) * (Sb9 / S0)**(n9 - n8)*(Sb8 / S0)**(n8 - n7) * (Sb7 / S0)**(n7 - n6) * (Sb6 / S0)**(n6 - n5) * (Sb5 / S0)**(n5 - n4) * (Sb4 / S0)**(n4 - n3)*(Sb3 / S0)**(n3 - n2) * (Sb2 / S0)**(n2 - n1) * (S / S0)**(n1)
    elif Sb2 < S <= Sb3:
        result = As * (Sb12 / S0)**(n12 - n11) * (Sb11 / S0)**(n11 - n10) * (Sb10 / S0)**(n10 - n9) * (Sb9 / S0)**(n9 - n8)*(Sb8 / S0)**(n8 - n7) * (Sb7 / S0)**(n7 - n6) * (Sb6 / S0)**(n6 - n5) * (Sb5 / S0)**(n5 - n4) * (Sb4 / S0)**(n4 - n3)*(Sb3 / S0)**(n3 - n2) * (S / S0)**(n2)
    elif Sb3 < S <= Sb4:
        result = As * (Sb12 / S0)**(n12 - n11) * (Sb11 / S0)**(n11 - n10) * (Sb10 / S0)**(n10 - n9) * (Sb9 / S0)**(n9 - n8)*(Sb8 / S0)**(n8 - n7) * (Sb7 / S0)**(n7 - n6) * (Sb6 / S0)**(n6 - n5) * (Sb5 / S0)**(n5 - n4) * (Sb4 / S0)**(n4 - n3)*(S / S0)**(n3)
    elif Sb4 < S <= Sb5:
        result = As * (Sb12 / S0)**(n12 - n11) * (Sb11 / S0)**(n11 - n10) * (Sb10 / S0)**(n10 - n9) * (Sb9 / S0)**(n9 - n8)*(Sb8 / S0)**(n8 - n7) * (Sb7 / S0)**(n7 - n6) * (Sb6 / S0)**(n6 - n5) * (Sb5 / S0)**(n5 - n4) * (S / S0)**(n4)
    elif Sb5 < S <= Sb6:
        result = As * (Sb12 / S0)**(n12 - n11) * (Sb11 / S0)**(n11 - n10) * (Sb10 / S0)**(n10 - n9) * (Sb9 / S0)**(n9 - n8)*(Sb8 / S0)**(n8 - n7) * (Sb7 / S0)**(n7 - n6) * (Sb6 / S0)**(n6 - n5) * (S / S0)**(n5)
    elif Sb6 < S <= Sb7:
        result = As * (Sb12 / S0)**(n12 - n11) * (Sb11 / S0)**(n11 - n10) * (Sb10 / S0)**(n10 - n9) * (Sb9 / S0)**(n9 - n8)*(Sb8 / S0)**(n8 - n7) * (Sb7 / S0)**(n7 - n6) * (S / S0)**(n6)
    elif Sb7 < S <= Sb8:
        result = As * (Sb12 / S0)**(n12 - n11) * (Sb11 / S0)**(n11 - n10) * (Sb10 / S0)**(n10 - n9) * (Sb9 / S0)**(n9 - n8)*(Sb8 / S0)**(n8 - n7) * (S / S0)**(n7)
    elif Sb8 < S <= Sb9:
        result = As * (Sb12 / S0)**(n12 - n11) * (Sb11 / S0)**(n11 - n10) * (Sb10 / S0)**(n10 - n9) * (Sb9 / S0)**(n9 - n8)*(S / S0)**(n8)
    elif Sb9 < S <= Sb10:
        result = As * (Sb12 / S0)**(n12 - n11) * (Sb11 / S0)**(n11 - n10) * (Sb10 / S0)**(n10- n9) * (S / S0)**(n9)
    elif Sb10 < S <= Sb11:
        result = As * (Sb12 / S0)**(n12 - n11) * (Sb11 / S0)**(n11 - n10) * (S / S0)**(n10)
    elif Sb11 < S <= Sb12:
        result = As * (Sb12 / S0)**(n12 - n11) * (S / S0)**(n11)
    elif Sb12 < S <= Sb13:
        result = As * (S / S0)**(n12)
    elif Sb13 < S <= Sb14:
        result = As * (Sb13 / S0)**(n12 - n13) * (S / S0)**(n13)
    elif Sb14 < S <= Sb15:
        result = As * (Sb13 / S0)**(n12 - n13) * (Sb14 / S0)**(n13 - n14) * (S / S0)**(n14)
    elif Sb15 < S <= Sb16:
        result = As * (Sb13 / S0)**(n12 - n13) * (Sb14 / S0)**(n13 - n14) * (Sb15 / S0)**(n14 - n15) * (S / S0)**(n15)
    elif Sb16 < S <= Sb17:
        result = As * (Sb13 / S0)**(n12 - n13) * (Sb14 / S0)**(n13 - n14) * (Sb15 / S0)**(n14 - n15) * (Sb16 / S0)**(n15 - n16) * (S / S0)**(n16)
    elif Sb17 < S <= Sb18:
        result = As * (Sb13 / S0)**(n12 - n13) * (Sb14 / S0)**(n13 - n14) * (Sb15 / S0)**(n14 - n15) * (Sb16 / S0)**(n15 - n16)*(Sb17 / S0)**(n16 - n17) * (S / S0)**(n17)
    elif Sb18 < S <= Sb19:
        result = As * (Sb13 / S0)**(n12 - n13) * (Sb14 / S0)**(n13 - n14) * (Sb15 / S0)**(n14 - n15) * (Sb16 / S0)**(n15 - n16)*(Sb17 / S0)**(n16 - n17) * (Sb18 / S0)**(n17 - n18) * (S / S0)**(n18)
    elif Sb19 < S <= Sb20:
        result = As * (Sb13 / S0)**(n12 - n13) * (Sb14 / S0)**(n13 - n14) * (Sb15 / S0)**(n14 - n15) * (Sb16 / S0)**(n15 - n16)*(Sb17 / S0)**(n16 - n17) * (Sb18 / S0)**(n17 - n18) * (Sb19 / S0)**(n18 - n19) * (S / S0)**(n19)
    elif Sb20 < S <= Sb21:
        result = As * (Sb13 / S0)**(n12 - n13) * (Sb14 / S0)**(n13 - n14) * (Sb15 / S0)**(n14 - n15) * (Sb16 / S0)**(n15 - n16)*(Sb17 / S0)**(n16 - n17) * (Sb18 / S0)**(n17 - n18) * (Sb19 / S0)**(n18 - n19) * (Sb20 / S0)**(n19 - n20) * (S / S0)**(n20)
    elif Sb21 < S <= Sb22:
        result = As * (Sb13 / S0)**(n12 - n13) * (Sb14 / S0)**(n13 - n14) * (Sb15 / S0)**(n14 - n15) * (Sb16 / S0)**(n15 - n16)*(Sb17 / S0)**(n16 - n17) * (Sb18 / S0)**(n17 - n18) * (Sb19 / S0)**(n18 - n19) * (Sb20 / S0)**(n19 - n20) * (Sb21 / S0)**(n20 - n21) * (S / S0)**(n21)
    elif Sb22 < S <= Sb23:
        result = As * (Sb13 / S0)**(n12 - n13) * (Sb14 / S0)**(n13- n14) * (Sb15 / S0)**(n14 - n15) * (Sb16 / S0)**(n15 - n16)*(Sb17 / S0)**(n16 - n17) * (Sb18 / S0)**(n17 - n18) * (Sb19 / S0)**(n18 - n19) * (Sb20 / S0)**(n19 - n20) *(Sb21 / S0)**(n20 - n21) * (Sb22 / S0)**(n21 - n22) * (S / S0)**(n22)
    elif Sb23 < S <= Sb24:
        result = As * (Sb13 / S0)**(n12 - n13) * (Sb14 / S0)**(n13 - n14) * (Sb15 / S0)**(n14 - n15) * (Sb16 / S0)**(n15 - n16)*(Sb17 / S0)**(n16 - n17) * (Sb18 / S0)**(n17 - n18) * (Sb19 / S0)**(n18 - n19) * (Sb20 / S0)**(n19 - n20) *(Sb21 / S0)**(n20 - n21) * (Sb22 / S0)**(n21 - n22) * (Sb23 / S0)**(n22 - n23) * (S / S0)**(n23)
    elif Sb24 < S <= Sb25:
        result = As * (Sb13 / S0)**(n12 - n13) * (Sb14 / S0)**(n13 - n14) * (Sb15 / S0)**(n14 - n15) * (Sb16 / S0)**(n15 - n16)*(Sb17 / S0)**(n16 - n17) * (Sb18 / S0)**(n17 - n18) * (Sb19 / S0)**(n18- n19) * (Sb20 / S0)**(n19 - n20) *(Sb21 / S0)**(n20 - n21) * (Sb22 / S0)**(n21 - n22) * (Sb23 / S0)**(n22 - n23) * (Sb24 / S0)**(n23 - n24) * (S / S0)**(n24)
    else:
        result = As * (Sb13 / S0)**(n12 - n13) * (Sb14 / S0)**(n13 - n14) * (Sb15 / S0)**(n14 - n15) * (Sb16 / S0)**(n15 - n16)*(Sb17 / S0)**(n16 - n17) * (Sb18 / S0)**(n17 - n18) * (Sb19 / S0)**(n18 - n19) * (Sb20 / S0)**(n19 - n20)*(Sb21 / S0)**(n20 - n21) * (Sb22 / S0)**(n21 - n22) * (Sb23 / S0)**(n22 - n23) * (Sb24 / S0)**(n23 - n24)*(Sb25 / S0)**(n24 - n25) * (S / S0)**(n25)
    return result
