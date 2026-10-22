# Changements par rapport à la version soumise à OJMO

Guide de relecture de `sn-preprint-theorie-diff.pdf` (65 pages) : ajouts en bleu souligné,
suppressions en rouge barré. Les numéros de page renvoient à ce PDF.

**À savoir avant de lire**

- Les figures 2 à 6 ont été régénérées à partir de la nouvelle campagne (21 octobre 2026,
  code de `3f0744f`, tag `coap-2026`). Le diff ne marque pas les figures, seulement leurs
  légendes.
- Une équation modifiée est remplacée en bloc : l'ancienne en rouge, la nouvelle en bleu.
- Le §3.1 et l'annexe A sont largement réécrits et apparaissent en longs blocs rouges puis bleus.
- Quelques renvois du texte supprimé s'affichent « ?? » : leurs étiquettes n'existent plus.

**Par section**

- **Titre, résumé et introduction (p. 1–6).** Dans le titre et les mots-clés, « least-squares »
  employé comme nom devient « least squares ». Résumé réécrit. Contributions reformulées (p. 5) :
  - l'équation sécante structurée n'est vraie qu'au second ordre près ;
  - la règle hybride fondée sur le lagrangien augmenté est attribuée à Tjoa et Biegler ;
  - solveur interne explicite sur le polyèdre, qui prolonge Conn et al. (1988) ;
  - analyse externe reprise de Conn, Gould, Sartenaer et Toint (1996) ;
  - code au tag `coap-2026`.
- **§2.1 (p. 7–12).**
  - Le test de criticité de l'algorithme 1 utilise le cône tangent des bornes ε-actives, avec
    ε_K = min(ε_max, ω_K), comme dans Conn et al. (1996) ; μ₀ > 1 exigé.
  - L'algorithme 1 renvoie λ̄_K, et repart de x_K après une itération externe ratée.
- **§2.2–2.3 (p. 12–17).**
  - Algorithme 2 : H_{k+1} = H_k si le pas est rejeté ; le rayon suit la règle de la §2.5, celle
    qu'utilise la preuve du théorème 5.
  - Le chemin de Cauchy suit des faces emboîtées, à partir d'un ensemble de travail choisi parmi
    les bornes ε-actives selon le signe des multiplicateurs ; pour une boîte avec ε = 0, c'est le
    chemin du gradient projeté.
- **§2.4 (p. 17–21).**
  - Remarque nouvelle (p. 18–20) : imposer l'équation sécante standard (choix de Dennis et Walker)
    introduit l'écart (32), pondéré par μ. La comparaison sur les 82 instances l'appuie : de 7 % à
    57 % d'évaluations de résidus en plus, et SR1 échoue sur le problème 5.15.
  - La règle hybride (35) est écrite sous forme multiplicative, qui reste valable quand le
    lagrangien augmenté est négatif.
- **§2.5 (p. 21–25).** Seuil ε_max ≤ ε̄ (lemme 1). Multiplicateurs de moindre violation de signe
  quand des bornes actives sont redondantes. Garde de SR1 avec κ_sds = 10⁻⁸. Initialisation de
  BFGS. Cas des bornes seules (p. 23) : ε = 0, et lien entre la mesure du gradient projeté et le
  test (8).
- **§3.1 (p. 25–38), à relire en priorité.** Réécrit pour les bornes ε-actives :
  - lemme 1 (p. 28) : le seuil ε̄ ne dépend que de Ω ;
  - proposition 2 (p. 30) : propriétés du chemin calculé ;
  - lemme 3 (p. 32) : ‖z₀‖ ≥ κ₀ π_ε ;
  - théorème 4 (p. 34) : décroissance de Cauchy ;
  - théorème 5 (p. 36) : convergence des itérations internes sous la seule hypothèse 6.

  La remarque de la page 34 montre que, sans le seuil ε, le chemin ne garantit pas la
  décroissance. L'ancienne relation (38), qui était fausse, a été retirée.
- **§3.2 (p. 38–44).** Présentée comme l'analyse de Conn et al. (1996), dont le test de criticité
  est repris. Lemme 10 (p. 43) : ce test certifie la mesure ‖P_Ω(x − ∇Φ) − x‖. La théorie couvre
  l'algorithme 1 avec l'algorithme 2.
- **§4 (p. 44–54).** Tous les chiffres, figures et tables viennent de la nouvelle campagne.
  - Introduction (p. 44–45) : seules cinq des 82 instances ont une contrainte linéaire explicite ;
    les tests d'arrêt de l'implémentation sont relatifs, ce que l'analyse ne couvre pas ;
    paramètres alignés sur le code (τ = 10, η₁ = 0,05, η₂ = 0,9).
  - §4.1.1 (p. 46–47) : BFGS n'échoue plus, et l'échec de SR1 sur HS373 est expliqué.
  - §4.1.2 (p. 47–51) : strates 18/16/23/22 et table 1 mise à jour. Le seuil de rentabilité est
    donné par ses quartiles (0,003 et 1,18 ms), et non plus par sa médiane, trop instable.
  - §4.2 (p. 51–53) : rapports avec IPOPT mis à jour. Seul Percival avait une limite de temps.
    Percival est comparé sur les 71 instances où il converge (4,28 / 2,94 / 1,47) ; ses deux
    dépassements de temps sont décrits à part.
  - §4.3, nouveau (p. 53–54) : contraintes linéaires gardées explicites ou pénalisées, sur
    15 problèmes (table 2).
  - L'ancienne section « Reproducibility » est retirée (p. 53) : ce qu'elle disait figure dans la
    déclaration « Data availability », et les scripts sont décrits dans la documentation du code.
- **Conclusion (p. 54).** Inchangée, à une précision près (« to our knowledge »).
- **Remerciements et déclarations (p. 54–55).** Remerciements à Mitacs et à l'IVADO. Nouvelle
  section « Statements and Declarations » : financement, intérêts concurrents, disponibilité
  des données et du code.
- **Annexe A (p. 55–58).** Calcul du point de Cauchy sur des ensembles de travail emboîtés, avec
  le choix de l'ensemble initial décrit au §2.3 ; algorithme 4 corrigé (test d'arrêt
  T(t_{i+1}) = {0}).
- **Bibliographie (p. 59–65).** Une vingtaine de références corrigées après vérification sur
  les sources. Neuf références ajoutées : Bertsekas (1982), Calamai et Moré (1987), Dennis et
  Walker (1981), Hoffman (1952), McCormick (1969), Rosen (1960), Topkis et Veinott (1967),
  Wolfe (1972), Zoutendijk (1960).
