# Changements par rapport à la version soumise à OJMO

Guide de relecture de `sn-preprint-theorie-diff.pdf` (63 pages) : ajouts en bleu souligné,
suppressions en rouge barré. Les numéros de page renvoient à ce PDF.

**À savoir avant de lire**

- Les figures 1 à 15 ont été régénérées à partir de la nouvelle campagne (21 octobre 2026,
  code de `3f0744f`, tag `coap-2026`). Le diff ne marque pas les figures, seulement leurs
  légendes.
- Une équation modifiée est remplacée en bloc : l'ancienne en rouge, la nouvelle en bleu.
- Le §3.1 et l'annexe A sont largement réécrits et apparaissent en longs blocs rouges puis bleus.
- Quelques renvois du texte supprimé s'affichent « ?? » : leurs étiquettes n'existent plus.

**Par section**

- **Résumé et introduction (p. 1–6).** Résumé réécrit. Contributions reformulées (p. 5) :
  l'équation sécante structurée n'est vraie qu'au second ordre près ; solveur interne
  explicite sur le polyèdre, qui prolonge Conn et al. (1988) ; analyse externe reprise de
  Conn, Gould, Sartenaer et Toint (1996) ; code au tag `coap-2026`.
- **§2.1 (p. 7–10).** Le test de criticité de l'algorithme 1 utilise le cône tangent des bornes
  ε-actives, avec ε_K = min(ε_max, ω_K), comme dans Conn et al. (1996) ; μ₀ > 1 exigé.
- **§2.2–2.3 (p. 11–17).** Algorithme 2 : H_{k+1} = H_k si le pas est rejeté. Le chemin de
  Cauchy suit des faces emboîtées, à partir d'un ensemble de travail choisi parmi les bornes
  ε-actives selon le signe des multiplicateurs ; sur un polyèdre général, il peut différer du
  chemin du gradient projeté.
- **§2.4 (p. 17–20).** Remarque nouvelle (p. 18–20) : imposer l'équation sécante standard
  (choix de Dennis et Walker) introduit l'écart (32), pondéré par μ. La comparaison sur les
  82 instances l'appuie : de 7 % à 57 % d'évaluations de résidus en plus, et SR1 échoue sur
  le problème 5.15.
- **§2.5 (p. 20–24).** Seuil ε_max ≤ ε̄ (lemme 1). Multiplicateurs de moindre violation de signe
  quand des bornes actives sont redondantes. Garde de SR1 avec κ_sds = 10⁻⁸.
- **§3.1 (p. 24–37), à relire en priorité.** Réécrit pour les bornes ε-actives :
  - lemme 1 : le seuil ε̄ ne dépend que de Ω ;
  - proposition 2 : propriétés du chemin calculé ;
  - lemme 3 : ‖z₀‖ ≥ κ₀ π_ε ;
  - théorème 4 : décroissance de Cauchy ;
  - théorème 5 : convergence des itérations internes sous la seule hypothèse 6.

  La remarque de la page 33 montre que, sans le seuil ε, le chemin ne garantit pas la
  décroissance. L'ancienne relation (38), qui était fausse, a été retirée.
- **§3.2 (p. 37–43).** Présentée comme l'analyse de Conn et al. (1996), dont le test de criticité
  est repris. Lemme 10 (p. 42) : ce test certifie la mesure ‖P_Ω(x − ∇Φ) − x‖.
- **§4 (p. 43–53).** Tous les chiffres, figures et tables viennent de la nouvelle campagne.
  - Introduction (p. 43–44) : seules 5 des 82 instances ont une contrainte linéaire explicite ;
    tolérances relatives de l'implémentation ; paramètres alignés sur le code (τ = 10,
    η₁ = 0,05, η₂ = 0,9).
  - §4.1.1 (p. 44–45) : BFGS n'échoue plus, et l'échec de SR1 sur HS373 est expliqué.
  - §4.1.2 (p. 45–49) : strates 18/16/23/22 et table 1 mise à jour. Le seuil de rentabilité
    est donné par ses quartiles (0,003 et 1,18 ms), et non plus par sa médiane, trop instable.
  - §4.2 (p. 49–51) : rapports avec IPOPT mis à jour. Percival est comparé sur les 71 instances
    où il converge (4,28 / 2,94 / 1,47) ; ses deux dépassements de temps sont décrits à part.
  - §4.3, nouveau (p. 51–53) : contraintes linéaires gardées explicites ou pénalisées, sur
    15 problèmes (table 2).
  - §4.4 (p. 53) : tag `coap-2026` et scripts de toutes les expériences.
- **Conclusion (p. 53).** Inchangée.
- **Remerciements et déclarations (p. 53–54).** Remerciements à Mitacs et à l'IVADO. Nouvelle
  section « Statements and Declarations » : financement, intérêts concurrents, disponibilité
  des données et du code.
- **Annexe A (p. 54–56).** Calcul du point de Cauchy sur des ensembles de travail emboîtés, avec
  le choix de l'ensemble initial décrit au §2.3.
- **Bibliographie (p. 56–63).** Une vingtaine de références corrigées après vérification sur
  les sources. Neuf références ajoutées : Bertsekas (1982), Calamai et Moré (1987), Dennis et
  Walker (1981), Hoffman (1952), McCormick (1969), Rosen (1960), Topkis et Veinott (1967),
  Wolfe (1972), Zoutendijk (1960).
