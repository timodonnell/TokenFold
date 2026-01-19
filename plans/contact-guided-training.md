# Contact-Guided Structure Prediction

## Motivation

Kanzi tokens are arbitrary learned codes - the model must discover what they mean from scratch. Contacts provide semantic grounding by telling the model which residues must be spatially close.

## Approach

Use inter-residue contacts as part of training to help the model "understand" Kanzi tokens better.

### Contact Definition
- Contact = C-alpha distance < 8Å
- Only non-local contacts: |i - j| > 6 (local contacts are trivial)
- Extract from ground truth C-alpha coordinates

## Implementation Phases

### Phase 1: Contacts as Input Hints (Baseline) ✅ CURRENT

Format:
```
<AA> M K T L ... <CONTACTS> 5-20 7-35 10-45 ... <SEP> <KANZI> <K100> ...
```

- Provide ground truth contacts as input
- Model predicts Kanzi tokens conditioned on contacts
- Tests whether contact information helps Kanzi prediction
- Simple to implement, good baseline

### Phase 2: Predict Contacts Then Kanzi (Chain-of-Thought)

Format:
```
<AA> M K T L ... <SEP> <CONTACTS> 5-20 7-35 ... <SEP> <KANZI> <K100> ...
```

- Model first predicts contacts from sequence
- Then predicts Kanzi tokens using its own contact predictions
- Chain-of-thought reasoning about structure
- Leverages pretrained knowledge (numbers are standard tokens)

### Phase 3: Verify Contacts After Kanzi (Teach Kanzi Semantics)

Format:
```
<AA> M K T L ... <SEP> <KANZI> <K100> <K200> ... <SEP> <CONTACTS> 5-20 7-35 ...
```

- Model predicts Kanzi first
- Then must predict which contacts are satisfied
- Forces model to learn what Kanzi tokens MEAN
- Can't predict contacts without understanding 3D structure encoded in tokens

### Phase 4: Full Multi-Task System

Format:
```
<AA> M K T ... <PRED_CONTACTS> 5-20 7-35 ... <KANZI> <K100> ... <VERIFY> 5-20:yes 7-35:yes 12-80:no ...
```

Three tasks in one forward pass:
1. **Predict contacts from sequence** - forces learning of fold patterns
2. **Predict Kanzi from sequence + predicted contacts** - contacts guide generation
3. **Verify contacts from Kanzi** - teaches Kanzi semantics

The verify step is key: it forces the model to understand that <K100> at position 5 and <K200> at position 20 must produce coordinates that are close.

## Why This Leverages Pretraining

- Numbers "5", "20", "35" are standard tokens in pretrained vocabulary
- "contact", "close", "near" are English words the model knows
- The model understands numerical relationships from pretraining
- No new vocabulary needed (unlike Kanzi tokens)

## Alternative: Contact Consistency Loss

Instead of changing the output format, add auxiliary loss:
1. Model predicts Kanzi tokens as usual
2. Decode Kanzi → coordinates
3. Compute predicted contacts from coordinates
4. Loss = cross-entropy on contact prediction

Teaches Kanzi semantics through loss function rather than output format.

## Metrics

- RMSD (primary metric)
- Contact precision/recall (for phases 2-4)
- Token accuracy
- Fraction of predicted structures with RMSD < 2Å, < 4Å

## Questions to Answer

1. Does providing contacts as hints improve RMSD? (Phase 1)
2. Can the model learn to predict contacts from sequence? (Phase 2)
3. Does predicting contacts after Kanzi improve Kanzi quality? (Phase 3)
4. Does multi-task training help all tasks? (Phase 4)
