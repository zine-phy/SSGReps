# Run from this directory: gap -q -A -b construct.g
# Exact cyclotomic arithmetic throughout. No floating point eigensolver.
Read("group_data.g");;
SetAssertionLevel(1);;
start := Runtime();;
# Right regular permutations, so GAP multiplication has the same order as mt.
perms := List([1..h],g -> PermList(Concatenation(
  List([1..h],x -> mt[x][g]+h*ce[x][g]),
  List([1..h],x -> mt[x][g]+h*((1+ce[x][g]) mod 2)))));;
z := PermList(Concatenation([h+1..2*h],[1..h]));;
ext := Group(z);; generatorIndices := [];;
for i in [1..h] do
  if not perms[i] in ext then
    Add(generatorIndices,i);
    ext := Group(Concatenation([z],perms{generatorIndices}));
  fi;
od;
Assert(1,Size(ext)=2*h);
Print("Central extension order: ",Size(ext),"; H generators: ",generatorIndices,"\n");
classes := ConjugacyClasses(ext);;
classOf := List(perms,p -> PositionProperty(classes,c -> p in c));;
zclass := PositionProperty(classes,c -> z in c);;
characters := Filtered(Irr(ext),c -> c[zclass]=-c[1]);;
wigner := c -> Sum([h+1..n],x -> (-1)^ce[x][x]*c[classOf[mt[x][x]]])/h;;
Print("Physical H irreps (dimension, Wigner index): ",List(characters,c -> [c[1],wigner(c)]),"\n");
chi := First(characters,c -> c[1]=8 and wigner(c)=0);;
Assert(1,chi<>fail);
ConjMat := m -> List(m,row -> List(row,ComplexConjugate));;
AdjMat := m -> TransposedMat(ConjMat(m));;
# Find a linear character occurring once in chi|B, without enumerating the
# whole subgroup lattice. Randomness chooses finite subgroups, never numbers
# approximating the matrix entries. The seed makes the search reproducible.
Reset(GlobalMersenneTwister,2218363);;
theta := fail;; restricted := fail;;
for trial in [1..50] do
  B := Group(z); C := ext;
  while Size(C)>Size(B) do
    repeat t:=Random(C); until not t in B;
    B:=Group(Concatenation(GeneratorsOfGroup(B),[t]));
    C:=Centralizer(ext,B);
  od;
  tblB:=CharacterTable(B);;
  irrB:=Irr(tblB);;
  FusionConjugacyClasses(tblB,CharacterTable(ext));
  restricted:=Restricted(chi,tblB);;
  theta:=First(irrB,x -> x[1]=1 and ScalarProduct(restricted,x)=1);
  if theta<>fail then break; fi;
od;
Assert(1,theta<>fail);
rhoB:=IrreducibleRepresentationsDixon(B,theta);;
induced:=InducedRepresentation(rhoB,ext);;
ambientDimension:=Index(ext,B);;
Print("Inducing from subgroup of order ",Size(B)," into dimension ",ambientDimension,".\n");
# Exact central idempotent projects the induced module onto chi, with
# multiplicity one. Deterministic Gram-Schmidt keeps its basis fairly small.
projector:=NullMat(ambientDimension,ambientDimension);;
for i in [1..h] do
  projector:=projector+2*ComplexConjugate(chi[classOf[i]])*Image(induced,perms[i]);
od;
projector:=chi[1]/Size(ext)*projector;;
Assert(1,projector*projector=projector and AdjMat(projector)=projector);
basisRows:=[];;
classesB:=ConjugacyClasses(B);;
Print("Restriction multiplicities: ",List(irrB,x -> ScalarProduct(restricted,x)),"\n");
for thetaWeight in irrB do
 if ScalarProduct(restricted,thetaWeight)>0 then
  weightProjector:=Sum([1..Length(classesB)],i -> ComplexConjugate(thetaWeight[i])*Image(induced,Representative(classesB[i])))/Size(B);
  jointProjector:=projector*weightProjector;
  for col in TransposedMat(jointProjector) do
    v:=ShallowCopy(col);
    for b in basisRows do v:=v-(List(b,ComplexConjugate)*v)*b; od;
    vnorm:=List(v,ComplexConjugate)*v;
    if vnorm<>0 then
      Assert(1,IsRat(vnorm) and vnorm>0);
      Add(basisRows,v/(ER(NumeratorRat(vnorm))/ER(DenominatorRat(vnorm))));
    fi;
  od;
 fi;
od;
basis:=TransposedMat(basisRows);;
d:=Length(basisRows);;
Assert(1,d=8 and AdjMat(basis)*basis=IdentityMat(d));
D:=List(perms,p -> AdjMat(basis)*Image(induced,p)*basis);;
Assert(1,AdjMat(basis)*Image(induced,z)*basis=-IdentityMat(d));
Assert(1,List(D,TraceMat)=List(classOf,i -> chi[i]));
Print("Constructed exact 8D unitary-subgroup representation.\n");
Print("8D field conductor: ",Conductor(Set(Flat(D))),"; entry count: ",Length(Set(Flat(D))),"\n");
a := h+1;;
ainv := Position(mt[a],1);;
a2 := mt[a][a];;
block := function(x,y)
  local b,i,j;
  b:=NullMat(2*d,2*d);
  for i in [1..d] do for j in [1..d] do
    b[i][j]:=x[i][j]; b[i+d][j+d]:=y[i][j];
  od; od;
  return b;
end;;
partner := [];;
for i in [1..h] do
  j:=mt[mt[ainv][i]][a];
  partner[i]:=(-1)^(ce[i][a]-ce[a][j])*ConjMat(D[j]);
od;
M := List([1..h],i -> block(D[i],partner[i]));;
Print("Constructed unitary blocks.\n");
MA := NullMat(2*d,2*d);;
for i in [1..d] do for j in [1..d] do
  MA[i][j+d]:=(-1)^ce[a][a]*D[a2][i][j];
od; MA[i+d][i]:=1; od;
for i in [1..h] do M[mt[a][i]]:=(-1)^ce[a][i]*MA*ConjMat(M[i]); od;
Print("Constructed antiunitary blocks.\n");
Assert(1,M[a]=MA);
charsH := List(D,TraceMat);;
charsP := List(partner,TraceMat);;
normH := Sum(charsH,x -> x*ComplexConjugate(x))/h;;
overlap := Sum([1..h],i -> charsH[i]*ComplexConjugate(charsP[i]))/h;;
Assert(1,normH=1 and overlap=0 and wigner(chi)=0);
charsFull := List(M{[1..h]},TraceMat);;
normFull := Sum(charsFull,x -> x*ComplexConjugate(x))/h;;
antiSum := Sum([h+1..n],x -> (-1)^ce[x][x]*charsFull[mt[x][x]])/h;;
Assert(1,normFull=2 and antiSum=0 and (normFull+antiSum)/2=1);
Assert(1,ForAll(M,x -> x*AdjMat(x)=IdentityMat(2*d)));
Print("Exact unitarity and irreducibility PASS. Verifying all ",n*n," products...\n");
for i in [1..n] do
  for j in [1..n] do
    if i<=h then rhs:=M[j]; else rhs:=ConjMat(M[j]); fi;
    Assert(1,M[i]*rhs=(-1)^ce[i][j]*M[mt[i][j]]);
  od;
od;
Print("All exact semilinear products PASS.\n");
PrintTo("exact_matrices.g","# M[g] K^anti[g]; index matches group_data.json + 1\nM := ",M,";;\n");
PrintTo("exact_characters.g","charsH := ",charsH,";;\ncharsPartner := ",charsP,";;\n");
allEntries := Set(Flat(M));;
Print("Distinct entries: ",allEntries,"\n");
Print("Cyclotomic conductor: ",Conductor(allEntries),"\n");
Print("Total seconds: ",(Runtime()-start)/1000.,"\n");
jsonOut := OutputTextFile("exact_representation.json",false);;
SetPrintFormattingStatus(jsonOut,false);
PrintTo(jsonOut,"{\"ssg\":\"221.8.6.3\",\"dimension\":16,\"order\":",n,
  ",\"unitary_order\":",h,",\"extension_order\":",Size(ext),
  ",\"exact_products_checked\":",n*n,",\"unitary_checks\":",n,
  ",\"H_character_norm\":",normH,",\"partner_overlap\":",overlap,
  ",\"Wigner_index\":",wigner(chi),",\"corep_indicator\":",(normFull+antiSum)/2,
  ",\"generator_indices_1based\":",generatorIndices,
  ",\"conductor\":",Conductor(allEntries),",\"entries\":",List(allEntries,String),
  ",\"matrices\":",List(M,m -> List(m,row -> List(row,String))),
  ",\"H_characters\":",List(charsH,String),",\"partner_characters\":",List(charsP,String),"}\n");
CloseStream(jsonOut);
QUIT;
