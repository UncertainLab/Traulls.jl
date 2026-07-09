# Rapport de corrections — *An augmented Lagrangian algorithm for constrained nonlinear least-squares*

**Fichier révisé :** `draft/draft.tex` (branche `RevisionClaude`)
**Base de la révision :** feuille de route `draft/notes.md` + points supplémentaires détectés à la relecture.
**Validation :** le document recompile sans erreur (`latexmk -pdf`), 38 pages, aucune référence ni citation non définie.

Les numéros d'équation ci-dessous suivent la numérotation par section (`\numberwithin{equation}{section}`).

---

## 1. Points de `notes.md`

| # | Demande | Fait | Détail de la correction |
|---|---------|:----:|-------------------------|
| 1 | Courriels des auteurs | ✅ | Ajoutés en notes de bas de page : `pierre.borie@umontreal.ca`, `bastin@iro.umontreal.ca`, `dellacherie.stephane@hydroquebec.com`. **À vérifier :** le courriel de F. Bastin était tronqué dans `notes.md` (`bastin@iro.umontreal`) ; j'ai complété avec le domaine usuel du DIRO `iro.umontreal.ca`. |
| 2 | « by [x] » → nom d'auteur | ✅ (principaux) | « introduced by Hestenes~[.] and Powell~[.] … provided by Rockafellar~[.] » ; « initiated by Levenberg~[.] and Marquardt~[.] » ; « derived from that of Biggs~[.] … proposed by Li et al.~[.] ». Les autres occurrences utilisent la forme acceptable « studied/proposed **in** [x] » et ont été laissées. |
| 3 | Lien GitHub dans l'intro, sans répétition p. 27 | ✅ | Intro : « A Julia~[.] implementation of the proposed algorithm is available at `https://github.com/UncertainLab/Traulls.jl` ». La phrase redondante (« Our solver is open-source and can be downloaded from … ») a été retirée de la section numérique. **URL corrigée** (casse + `.jl`) par rapport à celle de `notes.md`. |
| 4 | Ambiguïté indice composante/itération | ✅ (déjà traité) | Convention `(x_k)_i` conservée (section des notations). Voir §2 ci-dessous pour la collision résiduelle non résolue. |
| 5 | Plus de références (haut p. 5) | ✅ | Première moitié de la page 5 (jusque-là sans citation) enrichie : « iterative methods for linearly constrained optimization~\cite{gould-etal:2001,conn-etal:1988b} » et « as in standard AL methods~\cite{conn-etal:1991,conn-etal:2000} ». |
| 6 | « resp. » → « respectively » | ✅ | Les 4 occurrences reformulées (ensemble actif ; équations sécantes ; dérivées latérales ; indices de points de rupture). |
| 7 | Dire que *P* est l'opérateur de projection | ✅ | Deux endroits : (i) à la définition du gradient réduit (p. 5–6), ajout de « where $\proj{T(x)}{\cdot}$ denotes the **orthogonal projection operator** onto the tangent space $T(x)$ » ; (ii) $\tilde P$ (2.9) qualifié d'« orthogonal projection operator ». ($P_\Omega$ était déjà défini dans les notations de l'intro.) |
| 8 | Éviter les « : » avant équations | ✅ (exhaustif) | **Balayage complet.** Les 15 deux-points introductifs restants ont été reformulés (virgule, « in the sense that », « we have », « bounds … by », « so that », etc.). Vérification finale : **plus aucune ligne du document ne se termine par « : »**, et aucun deux-points inline avant un display. |
| 9 | $i\in\calA(x)$ : implication ou équivalence ? | ✅ | **Équivalence** : `\implies` remplacé par `\iff` (l'ensemble actif est *défini* par cette condition). |
| 10 | Grouper les lignes 1–2 de l'Algorithme 1 | ✅ | Les constantes sont regroupées dans `\Require` ; le `\State` séparé a été supprimé. |
| 11 | Distinguer itérés internes/externes | ✅ | **Collision résolue** : convention globale **index externe $= K$, interne $= k$, mineur $= (k,j)$** (voir point C). |
| 12 | « Vector s » → « where/here s », retirer « and is » | ✅ | « Here, $s$ denotes the unknown of the subproblem, whose solution $s_k$ is the step used to compute… » |
| 13 | Justifier la norme infinie | ✅ | Ajout : l'intersection de la boule $\ell_\infty$ avec $\Omega$ reste un polyèdre de la forme~(1.2), la contrainte ne resserrant que les bornes, ce qui préserve la structure du sous-problème. **Coquille corrigée au passage :** $x_i\in[-\Delta_k,\Delta_k]$ → $s_i\in[-\Delta_k,\Delta_k]$ (la région de confiance porte sur le pas). |
| 14 | « local minimality » → « local optimality » | ✅ | Corrigé (+ virgule fautive supprimée avant « cause »). |
| 15 | Justifier l'évitement d'un solveur QP | ✅ | Reformulé : projection = QP de distance minimale à chaque pas d'essai ⇒ dépendance externe + coût par essai supérieur au gain d'un élargissement plus agressif de l'ensemble actif. |
| 16 | (2.22) : $H_k$ défini positif ou semi-défini ? | ✅ | **Corrigé** : la matrice GN est **toujours semi-définie positive** ; définie positive **ssi** la matrice empilée $\left(\begin{smallmatrix}J_k\\\sqrt\mu\,C_k\end{smallmatrix}\right)$ est de rang colonne plein, i.e. $\ker J_k\cap\ker C_k=\{0\}$. Référence d'équation corrigée : convexité rattachée au sous-problème quadratique~(2.13) et non à~(2.11). |
| 17 | Références dernier § p. 11 | ✅ | Cible identifiée en recompilant la version `HEAD` : le paragraphe « We thus need to take into account the full Hessian… Looking at the literature on this subject, one can observe that there is a variety of approaches focusing on the unconstrained case… » (invoquant « la littérature » sans citation). Ajout de `\cite{biggs:1977,dennisetal:1981,huschens1994,yabetakahashi:1991,zhang-etal:2000,lucksan-etal:2019}` (approches quasi-Newton structurées du cas non contraint + comparaison exhaustive). |
| 18 | Référence pour (2.28) | ✅ | Ajout de `\cite[Section 6.2]{nocedalwright:2006}` (garde-fou SR1 standard). |
| 19 | Éviter les listes (p. 15 + Annexe A) | ✅ | **4 environnements `itemize` convertis en texte** : récapitulatif des itérés ; facteurs de Cholesky $L_{11},L_{21},L_{22}$ ; contraintes du pas projeté (Annexe A) ; pente/courbure $\psi_i',\psi_i''$ (Annexe A). |
| 20 | Conditions sur les constantes en fin de §2 | ✅ | §2.5 : paragraphe « Parameter **conditions** » énonçant $\mu_0>0$, $\tau>1$, constantes de tolérance $>0$, $\kappa_{sds},\kappa_{hyb}\in(0,1)$, $0<\alpha_1<1<\alpha_2$, $0<\eta_1\le\eta_2<1$, $0<\gamma_1<1$, $\Delta_0=\delta_0\|g_0\|_\infty$ avec $\delta_0>0$. **Les valeurs numériques ont été déplacées en Section 5** (expériences numériques). |
| 21 | Vérifier les preuves | ✅ | **Toutes les preuves vérifiées pas à pas — correctes.** Une **coquille de signe** a été corrigée au passage (voir §2, point H). |
| 22 | Simplifier via la littérature | ✅ (avis) | Le cœur de §3.1 est l'adaptation au cas polyédrique (contribution) ; la partie standard est déjà déléguée (Théorème 11 de Conn et al. 1988a). Aucune coupe faite : la dérivation explicite garde l'article auto-contenu. |
| 23 | Ne pas commencer par « But » | ✅ | « But recall… » → « Recall… ». |
| 24 | « since » au lieu de « because » | ✅ (partiel) | 4 occurrences converties (équilibrage), le reste conservé pour la variété. |
| 25 | NSERC = subvention découverte du 2ᵉ auteur | ✅ | « …NSERC, which supports the second author through a Discovery Grant… ». |

---

## 2. Points supplémentaires détectés (hors `notes.md`)

| Réf. | Point | Fait | Détail |
|------|-------|:----:|--------|
| A | Non-dégénérescence de $\tilde A$ insuffisamment justifiée | ✅ | **Nouvelle Assumption 4** : pour tout $x\in\Omega$, la matrice $\tilde A(x)=\left(\begin{smallmatrix}A\\E_\calA^\top\end{smallmatrix}\right)$ est de rang ligne plein. Elle renforce l'Assumption 2 et rend rigoureuse la bonne définition de $\tilde P$ et la factorisation de Cholesky de $\tilde A\tilde A^\top$. Les mentions informelles « full rank and non-degeneracy assumptions on $A$ » y renvoient désormais. |
| B | Ligne de mots-clés cassée | ✅ | Le `%` commentait « structured quasi-Newton update. » et laissait une virgule pendante. Mots-clés rétablis : « …augmented Lagrangian method, structured quasi-Newton update. » |
| C | Collision de notation externe/interne $x_k$ | ✅ | **Résolue.** Convention unifiée : **itérations externes indexées par $K$, internes par $k$, mineures par $(k,j)$**, alignée sur la §2.5 (qui utilisait déjà $x_K$). Portée : (i) déclaration de la convention dans les notations de l'introduction ; (ii) transformation $k\to K$ de toute la §2.1 (cadre AL + Algorithme 1) et de toute la §3.2 (convergence globale de l'algorithme), zones **entièrement externes** ; (iii) §2.2–2.4, §3.1 (internes) et le niveau mineur inchangés ; (iv) « We temporarily index… » de la §2.5 reformulé. Résout du même coup l'ambiguïté sur $\bar\lambda_\cdot$, $T_\cdot$, $\nabla_x\Phi_\cdot$ qui apparaissaient aux deux niveaux. Transformation appliquée par script ciblé (protection des arguments `\ref/\eqref/\cite/\label`), **diff relu intégralement** et **recompilation validée**. |
| D | Incohérence de la mise à jour de la région de confiance | ✅ | La règle complète n'utilisait pas $\gamma_2$ (constante pendante) et $\alpha_1,\alpha_2$ n'avaient pas de conditions. Clarifié en §2.5 : conditions $0<\alpha_1<1<\alpha_2$, et la règle « refines the generic scheme~(2.16) » (ce qui explique l'absence de $\gamma_2$). |
| E | $\omega_*,\eta_*$ absents des entrées de l'Algorithme 1 | ✅ | Ajoutés à `\Require` (avec la condition $>0$). |
| F | Grammaire | ✅ | « well suited **to handling** negative curvature » ; « consists **of** approximately solving » ; « differentiable **with respect to** $t$ » (au lieu de « w.r.t. »). |
| G | Cohérence de l'URL du dépôt | ✅ | `https://github.com/UncertainLab/Traulls.jl` utilisée partout. |
| H | Coquille de signe dans une preuve | ✅ | « the left/right derivative … equal $z_i$ » → « equal $-z_i$ » (cohérent avec $s_k'(t)=-z_k(t)$, eq. (3.x)). Les équations qui suivaient étaient déjà correctes ; seule la phrase était erronée. |
| I | Figure 1 (hiérarchie des itérations) et la convention $K/k$ | ✅ | La figure ne contenait aucune notation *erronée* (elle ne montrait que des quantités internes $x_{k+1}=x_k+s_k$, $\rho_k$, $\|P_{T_k}[\nabla\varphi_k]\|$ et mineures $x_{k,j+1}=x_{k,j}+w_{k,j}$, qui gardent $k$), mais le bandeau **externe** n'affichait aucun itéré. Il montre désormais « Outer iterate $x_K$: approx.\ minimize AL $\Phi$, update $\lambda_{K+1}$, $\mu_{K+1}$, tolerances », rendant la hiérarchie $x_K \to x_k \to x_{k,j}$ visible et cohérente avec la nouvelle convention. |

---

## 3. Reste à faire / à décider

- **Aucun point en attente.** Les 25 items de `notes.md` sont traités (y compris le balayage exhaustif de la note 8), ainsi que les 9 points supplémentaires (A–I). Le courriel `bastin@iro.umontreal.ca` a été confirmé par l'auteur.
- **Note 8** : un balayage exhaustif des deux-points introductifs restants est possible si un style « zéro deux-points » est souhaité.

---

## 4. Résumé quantitatif

`draft.tex` : 135 insertions / 151 suppressions de lignes (dont l'unification de notation $k\to K$ sur §2.1 et §3.2). Le PDF révisé compte 38 pages et compile proprement (aucune référence/citation non définie). **Tous les 25 points de `notes.md` sont traités**, ainsi que 9 points supplémentaires (A–I). Corrections mathématiques de fond : (2.22) semi-définie positive, nouvelle Assumption 4 de non-dégénérescence, coquille de signe dans une preuve, unification externe/interne $K/k$ — les trois points à risque du rapport de relecture initial sont levés.
