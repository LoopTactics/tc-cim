#ifndef TACTICS_OPTIMIZER_H
#define TACTICS_OPTIMIZER_H

#include "tc/core/polyhedral/tactics/mapped_scop.h"

namespace tc {
namespace polyhedral {
namespace tactics {

isl::schedule optimizeGemmSchedule(const MappedScop& scop,
				   TacticsReplacements& replacements);

} // namespace tactics
} // namespace polyhedral
} // namespace tc

#endif
