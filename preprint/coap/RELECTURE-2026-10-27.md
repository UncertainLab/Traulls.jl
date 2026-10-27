# Relecture finale du manuscrit TRAULLS

27 octobre 2026 · Fabian Bastin

## Contexte

Cette relecture couvre tout le manuscrit COAP, en particulier les modifications de Pierre des 26 et 27 octobre (commits `f08ede5`, `7466fd4` et `1fdad8e`). Quatre corrections et deux coquilles ont été appliquées aux deux versions, `preprint/coap/sn-preprint-theorie.tex` et `preprint/main.tex`, dont les corps sont identiques au gabarit près.

Le texte ajouté apparaît en magenta dans les PDF (les liens et les citations sont déjà en bleu dans le gabarit Springer), grâce à une macro `\rev` définie dans le préambule des deux fichiers. Le texte supprimé n'apparaît plus : chaque section ci-dessous donne l'ancienne et la nouvelle version. Pour retirer la couleur avant la soumission, il suffit de remplacer la définition de la macro par `\newcommand{\rev}[1]{#1}`.

Les numéros d'équations, de théorèmes et de pages sont ceux de la version COAP ; les numéros de ligne sont donnés pour les deux fichiers.

## Bilan de la vérification

Aucun résultat faux n'a été trouvé : la correction de la semaine dernière tient, et les preuves remaniées par Pierre sont justes. Les quatre corrections portent sur la cohérence entre l'algorithme, le code et les énoncés, pas sur un résultat.

| Partie | Points vérifiés pas à pas |
| ------- | ------------------- |
| Lemme 1 (seuil ε̄) | programme linéaire définissant ρ\_S, définition de ε̄, points (i) à (iii) |
| Cônes T\_ε et N\_ε | polarité (Farkas), décomposition de Moreau (16) |
| Proposition 2 | chemin linéaire par morceaux, emboitement, arrêt de l'itération (44) |
| Lemme 3, partie corrigée | identités de Gram (47), G(F′) définie positive, itération (44) écrite en ϑ, récurrence descendante, constantes κ₁ et κ₀, F non vide |
| Théorème 4 | t₁ ≥ min(ε, Δ\_k)/‖z₀‖ et borne (49) |
| Théorème 5 | constantes c, ζ, c₁, c₂, c₃ et les trois étapes, sous réserve des corrections 2 et 3 |
| Lemmes 6 à 8, théorème 9 | inclusion des bornes ε-actives dans les bornes actives en x\_\*, multiplicateurs, cas μ borné ou non |
| Lemme 10 | les deux inégalités de certification et le cas ε = 0 |
| §2.4 | identité exacte de y\_k − ỹ\_k et développement (35), refait |
| §2.5, bornes seules | certification du test (8), composante par composante |
| §4 | table 1, 31 % = 15/48, 2/3 = 32/48, moyennes géométriques de la table 2 (40 % et 30 %) |

Les deux versions ont le même corps, au gabarit près, et compilent sans référence ni citation indéfinie.

## Correction 1 : règle hybride (38)

La règle (38) revient à la forme multiplicative, la seule qui reste définie et conforme au code quand le lagrangien augmenté est négatif ou nul. Emplacement : §2.4, p. 21 (COAP l. 656 et 660, `main.tex` l. 604 et 608).

**Problème.** Le commit `f08ede5` a remplacé le test multiplicatif par le rapport ζ\_k ≤ κ\_hyb et supprimé la phrase sur le cas φ(x\_k) ≤ 0. Or φ peut être négatif ou nul, à cause du terme ⟨λ, c(x)⟩. Si φ(x\_k) = 0, ζ\_k n'est pas défini ; si φ(x\_k) < 0, une descente donne ζ\_k < 0 et active la correction structurée, soit la décision inverse.

**Code.** `src/hessian.jl`, l. 455, retient l'approximation GN dès que `fx_prev - fx_new + delta_red > 0.1 * fx_prev`. C'est la forme multiplicative, au terme de sécurité `delta_red` près, de l'ordre de la précision machine.

**En pratique.** Le cas φ(x\_k) < 0 survient souvent dans les expériences de l'article, rejouées avec un enregistrement de chaque test (φ(x\_k) = 0 n'arrive jamais). Sur chacun de ces tests, le code a retenu GN, là où le rapport aurait activé la correction structurée.

| Expérience | Tests | φ(x\_k) < 0 | Instances concernées |
| --- | --- | --- | --- |
| §4.1-4.2, Hybrid-SR1 | 2 879 | 229 (8 %) | 30 sur 82 |
| §4.1, Hybrid-BFGS | 58 635 | 233 | 25 sur 82 |
| §4.3, formulation E | 148 | 0 | 0 sur 15 |
| §4.3, formulation P | 209 | 2 | 2 sur 15 (HS21, HS49) |

Les cas les plus nombreux viennent des problèmes 5.11, 5.14, 5.15 et 5.16 de Lukšan et Vlček, avec φ(x\_k) jusqu'à −1,7 × 10⁷ dès μ = 10. Les 194 exécutions reproduisent les compteurs publiés, sauf deux (Hybrid-BFGS sur 5.2 avec n = 1000, Hybrid-SR1 sur 5.12 avec n = 1000), sans doute à cause de la version de Julia (1.12.6 au lieu de 1.12.1).

Le terme ⟨λ, c(x)⟩ domine donc dès les premières itérations externes, et pas seulement près d'une solution à résidu nul. Ces chiffres ont été obtenus en redéfinissant la méthode `update_hessian!` des approximations hybrides, comme le fait `benchmark/secant_comparison/run_secant_comparison.jl`, avec les paramètres de la campagne. Par choix des auteurs, ils ne figurent pas dans l'article.

**Avant** (première ligne de (38)) :

$$
\zeta_k \le \kappa_{\mathrm{hyb}}
$$

**Après**, comme dans la version du 21 octobre (`592160d`) :

$$
\varphi(x_k)-\varphi(x_{k+1}) \le \kappa_{\mathrm{hyb}}\,\varphi(x_k)
$$

La phrase supprimée est rétablie après (38) : « When φ(x\_k) > 0, the test reads ζ\_k ≤ κ\_hyb; when φ(x\_k) ≤ 0, which can happen because of the term ⟨λ, c(x)⟩, any decrease of φ selects the GN approximation. » La définition de ζ\_k, juste avant (38), est conservée, car elle motive la règle et sert dans cette phrase.

## Correction 2 : énoncé du théorème 5

L'énoncé du théorème 5 précise désormais la règle de mise à jour du rayon qu'utilise sa preuve, celle du §2.5. Emplacement : §3.1, p. 30 (COAP l. 910, `main.tex` l. 858).

**Problème.** Depuis `f08ede5`, l'algorithme 2 met à jour le rayon selon le schéma générique (23). La preuve du théorème 5 utilise la règle de l'implémentation, donnée au §2.5. Les deux règles ne coïncident pas :

- après un échec, la règle du §2.5 peut réduire le rayon à α₁‖s\_k‖∞ < γ₁Δ\_k, ce que (23) interdit ;
- après un succès avec ρ\_k ∈ \[η₁, η₂), (23) autorise une réduction jusqu'à γ₂Δ\_k, alors que le deuxième pas de la preuve suppose Δ\_{k+1} ≥ Δ\_k.

Le théorème énoncé pour l'algorithme 2 tel qu'il est écrit n'était donc pas celui que la preuve établit.

**Avant** : « If Algorithm 2 generates an infinite sequence of iterates, then »

**Après** : « If Algorithm 2, with the radius update rule of Section 2.5, generates an infinite sequence of iterates, then »

La phrase de la preuve « From now on, the radius is updated by the rule used in our implementation… » reste juste et n'a pas changé. Autre option : rétablir dans l'algorithme 2 la ligne de `592160d`, « Set Δ\_{k+1} by the update rule of Section 2.5 », au prix de l'algorithme générique voulu par Pierre. Le résultat resterait vrai avec (23), mais le deuxième pas de la preuve serait à réécrire : il faudrait montrer qu'un petit rayon donne ρ\_k ≥ η₂.

## Correction 3 : matrice inchangée après un pas rejeté

Le §3.1 précise de nouveau que H\_k ne change pas après un pas rejeté, ce dont la preuve du théorème 5 a besoin. L'algorithme 2 reste celui de Pierre. Emplacement : §3.1, p. 26 (COAP l. 796, `main.tex` l. 744).

**Problème.** Le commit `f08ede5` a retiré « and update H\_{k+1} » et « and H\_{k+1} ← H\_k » de l'algorithme 2, ainsi que la phrase du §3.1 qui le rappelait. Il ne reste que le §2.4 : la mise à jour « is still computed at every successful iteration », ce qui n'exclut pas une mise à jour après un échec.

**Rôle dans la preuve.** Le dernier pas du théorème 5 traite le cas d'un nombre fini de succès. Il utilise que H\_k, donc b\_k, est constant à partir d'un certain rang. Sinon b\_k pourrait croître sans borne, et la minoration Δ\_k ≥ c₃/b\_k ne contredirait plus Δ\_k → 0.

**Avant** : ni l'algorithme 2 ni le §3.1 n'énoncent cette propriété.

**Après** : au §3.1, après la description du modèle q\_k, une version reformulée de la phrase supprimée par f08ede5, qui énonce l'hypothèse puis sa justification : « We assume that the approximation is not modified after an unsuccessful iteration, that is, H\_{k+1} = H\_k when ρ\_k < η₁. This is the case in our implementation: the approximations of Section 2.4 are updated from the derivatives at the trial point x\_k + s\_k, which are only evaluated when the step is accepted. »

C'est ce que fait le code : `update_hessian!` n'est appelée que dans la branche du pas accepté (`src/solver.jl`, l. 501). La mise à jour après un pas accepté reste implicite dans la ligne 3 de l'algorithme 2, « Form the model m\_k ». Cette ligne ne dit pas que H\_k reste fixe après un rejet, d'où la phrase du §3.1. Autre option : mentionner H\_k dans les deux branches de l'algorithme 2, comme au 21 octobre.

## Correction 4 : contribution (ii)

La contribution (ii) rattache de nouveau l'absence d'hypothèse sur les signes des multiplicateurs à notre certification, et non au cadre de Conn, Gould, Sartenaer et Toint (1996). Emplacement : §1, p. 5 (COAP l. 228, `main.tex` l. 175).

**Problème.** Dans la phrase de `7466fd4`, le « but » laisse entendre que l'analyse de Conn et al. (1996) suppose les multiplicateurs de bornes du bon signe. Ce n'est pas le cas : leur test de criticité, notre test (8), n'en fait aucune. C'est le gradient réduit seul, comme dans MINOS ou LSNNO, qui exige ce signe ; notre mesure π\_ε s'en passe grâce au terme θ^w (lemme 10).

**Avant** : « Our analysis follows the framework of \[CGST96\], of which our problem is a particular case, but without any assumption on the signs of the bound multipliers. To that end, we show that the measure used in our implementation, based on the reduced gradient and on least-squares multiplier estimates, certifies the criticality test of \[CGST96\]. »

**Après** : « Our analysis follows the framework of \[CGST96\], of which our problem is a particular case. To apply it, we show that the measure used in our implementation, based on the reduced gradient and on least-squares multiplier estimates, certifies the criticality test of \[CGST96\] without any assumption on the signs of the bound multipliers. »

C'est le sens de la version du 21 octobre, où la mesure certifiait ce test « without any assumption on the signs of the bound multipliers ». La précision compte d'autant plus que les auteurs de cet article pourraient être sollicités comme rapporteurs.

## Coquilles corrigées

Deux coquilles ont été corrigées dans les deux versions ; elles sont aussi marquées en magenta.

| Emplacement | Avant | Après |
| --- | --- | --- |
| Preuve du lemme 3, p. 29 (COAP l. 870, `main.tex` l. 818) | is negative which, the diagonal term being positive, | is negative, which, the diagonal term being positive, |
| Phrase avant le théorème 5, p. 30 (COAP l. 908, `main.tex` l. 856) | the inner iterations procedure | the inner iteration procedure |

## Points restants

Les deux `.tex` modifiés et les deux PDF régénérés sont commités sur main. Les points suivants, hors du texte de l'article, restent à régler avant la soumission.

- [ ] Relire les passages en magenta, puis retirer la couleur : remplacer la définition de `\rev` par `\newcommand{\rev}[1]{#1}` (COAP l. 113, `main.tex` l. 50) et recompiler.
- [ ] Mettre à jour `coap/CHANGEMENTS.md` et régénérer `sn-preprint-theorie-diff.pdf`. Ils décrivent encore l'état du 21 octobre : contre-exemple de la p. 34, chiffres de 7 % à 57 % au §2.4, références Hoffman, Wolfe et McCormick, qui ne sont plus citées (71 entrées).
- [ ] Choisir pour la comparaison des équations sécantes : le §2.4 ne cite plus ses chiffres, mais `coap/README.md`, `docs/src/benchmarks.md` et `benchmark/paper_numbers.jl` les annoncent encore. Cette phrase était le seul appui empirique de la remarque, désormais présentée comme heuristique.
- [ ] Encodage : depuis `7466fd4`, le `.tex` COAP est en UTF-8 (U+FFFD dans le commentaire de la l. 28), contrairement à `coap/README.md`. Sans effet sur le PDF, comme le commentaire périmé « + Hoffman (1952) » de la l. 1459.
- [ ] Optionnel : au §4 (COAP l. 1146), le test final est décrit avec π\_ε\*. Pour les 77 instances en boîte, le code utilise en fait la norme du gradient projeté du §2.5.
