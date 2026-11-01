#!/usr/bin/env perl
use strict;
use warnings;
use Digest::SHA qw(sha256_hex);
use JSON::PP;

sub read_bytes {
    open my $file, '<', $_[0] or die "$!";
    local $/;
    return <$file>;
}

my ($sass_path) = @ARGV;
die "provide cuobjdump SASS path\n" unless defined $sass_path;
my $reference = decode_json(read_bytes('docs/experiments/2026-10-28-draft-compiled-audit.json'));
my @expected = (@{$reference->{original_entries}}, $reference->{candidate});
my (%instructions, $name);
for my $line (split /\n/, read_bytes($sass_path)) {
    $name = $1 if $line =~ /Function : (\S+)/;
    next unless defined $name;
    while ($line =~ /0x([a-f0-9]{16})/g) {
        push @{$instructions{$name}}, "0x$1\n";
    }
}
die 'kernel coverage differs' unless @expected == 138 && keys(%instructions) == 138;
for my $entry (@expected) {
    die "missing $entry->{name}" unless exists $instructions{$entry->{name}};
    my $words = $instructions{$entry->{name}};
    die "changed $entry->{name}" unless @$words == $entry->{words}
        && sha256_hex(join '', @$words) eq $entry->{sha256};
}
print JSON::PP->new->canonical->pretty->encode({
    is_passed => JSON::PP::true, unchanged_entries => 138,
    reference => 'docs/experiments/2026-10-28-draft-compiled-audit.json',
    sass_sha256 => sha256_hex(read_bytes($sass_path)),
    scope => 'Instruction identity only; no achieved occupancy or traffic claim'
});
